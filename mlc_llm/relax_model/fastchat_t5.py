
from tvm import te, tir
from tvm.relax.op import (
    astype,
    matmul,
    maximum,
    permute_dims,
    reshape, 
    squeeze
)
from tvm.relax.op.nn import gelu_tanh, softmax

from .modules import (
    Linear,
    Embedding
)

class T5Config:
    def __init__(
        self,
        is_decoder
        vocab_size=32110,
        d_model=2048,
        d_kv=64,
        d_ff=5120,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=32,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        self.is_decoder = is_decoder
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        
        self.pad_token_id=pad_token_id,
        self.eos_token_id=eos_token_id,
        

class T5LayerNorm(nn.Module):
    # Equivlent to LlamaRMSNorm
    def __init__(self, hidden_size, dtype, eps=1e-6):
        self.weight = nn.Parameter((hidden_size,), dtype=dtype, name="t5_norm_weight")
        self.variance_epsilon = tvm.tir.const(eps, dtype)

    def forward(self, hidden_states):

        def f_rms_norm(x, weight):
            is_float32 = x.dtype == "float32"

            def f_square(x):
                return (
                    tir.Cast("float32", x) * tir.Cast("float32", x)
                    if not is_float32
                    else x * x
                )

            k = te.reduce_axis((0, x.shape[2]), name="k")
            square_sum = te.compute(
                (x.shape[0], x.shape[1]),
                lambda bsz, i: te.sum(f_square(x[bsz, i, k]), axis=k),
                name=x.op.name + "red_temp",
            )

            def f_div_cast(bsz, i, k):
                x_val = x[bsz, i, k]
                if not is_float32:
                    x_val = tir.Cast("float32", x_val)
                return x_val / tir.sqrt(
                    square_sum[bsz, i] / x.shape[2] + self.variance_epsilon
                )

            def f_mul_cast(x, y):
                value = x * y
                if not is_float32:
                    value = tir.Cast(x.dtype, value)
                return value

            return te.compute(
                x.shape,
                lambda bsz, i, k: f_mul_cast(weight(k), f_div_cast(bsz, i, k)),
                name="t5_norm",
            )

        return nn.emit_te(
            f_rms_norm, hidden_states, self.weight, primfunc_name_hint="t5_norm"
        )

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.dtype = config.dtype
        self.wi_0 = Linear(config.d_model, config.d_ff, self.dtype, bias=False)
        self.wi_1 = Linear(config.d_model, config.d_ff, self.dtype, bias=False)
        self.wo = Linear(config.d_ff, config.d_model, self.dtype, bias=False)

    def forward(self, hidden_states):
        hidden_gelu = gelu_tanh(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear

        # TODO(Dacheng): This seems problematic when loading with quantization.
        # Info: https://github.com/huggingface/transformers/issues/20287
        if hidden_states.struct_info.dtype != self.wo.struct_info.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.wo.struct_info.dtype))
        hidden_states = self.wo(hidden_states)
        return hidden_states

class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.dtype = config.dtype

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = Linear(self.d_model, self.inner_dim, self.dtype, bias=False)
        self.k = Linear(self.d_model, self.inner_dim, self.dtype, bias=False)
        self.v = Linear(self.d_model, self.inner_dim, self.dtype, bias=False)
        self.o = Linear(self.inner_dim, self.d_model, self.dtype, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(self.relative_attention_num_buckets, self.n_heads, self.dtype)

        
    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        key_value_states: relax.Expr = None,
        position_bias: relax.Expr = None,
        past_key_value: Optional[Tuple[relax.Expr, relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Union[Tuple[None, None], Tuple[relax.Expr, relax.Expr]]]:

        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        
        # Write out whether this is self or cross attention here for better code readability
        is_self_attn = key_value_states is None
        # hidden_states: [batch_size, seq_len, hidden_size]
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        batch_size, seq_len, _ = hidden_states.struct_info.shape
        kv_seq_len = all_seq_len_shape.struct_info.values[0]

        def _project(hidden_states, proj):
            return nn.emit(
                reshape(
                    proj(hidden_states),
                    (batch_size, seq_len, self.num_heads, self.head_dim),
                )
            )
        
        # q/k/v states: [batch_size, seq_len, num_attention_heads, head_size]
        q = _project(hidden_states, self.q_proj)

        # Handle self/cross attention logic
        if is_self_attn:
            k = _project(hidden_states, self.k_proj)
            v = _project(hidden_states, self.v_proj)
        elif:
            k = _project(key_value_states, self.k_proj)   
            v = _project(key_value_states, self.v_proj)   

        if past_key_value is not None:
            f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
            f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
            k_cache, v_cache = past_key_value
        
            # TODO(Dacheng): if self-attn, append; else, do nothing
            if is_self_attn: 
                k_cache = nn.emit(
                    relax.Call(
                        f_kv_cache_append,
                        args=[k_cache, squeeze(k, axis=0)],
                        sinfo_args=[relax.ObjectStructInfo()],
                    )
                )
                v_cache = nn.emit(
                    relax.Call(
                        f_kv_cache_append,
                        args=[v_cache, squeeze(v, axis=0)],
                        sinfo_args=[relax.ObjectStructInfo()],
                    )
                )
               
            batch_size, _, num_heads, head_size = k.struct_info.shape
            kv_cache_shape = R.shape([kv_seq_len, num_heads, head_size])
            kv_states_shape = R.shape([batch_size, kv_seq_len, num_heads, head_size])
            k = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[k_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, k.struct_info.dtype)],
                )
            )
            v = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[v_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, v.struct_info.dtype)],
                )
            )
            k = nn.emit(reshape(k, kv_states_shape))
            v = nn.emit(reshape(v, kv_states_shape))
            past_key_value = (k_cache, v_cache)
        else:
            past_key_value = (None, None)

        q = nn.emit(permute_dims(q, [0, 2, 1, 3]))
        k = nn.emit(permute_dims(k, [0, 2, 1, 3]))
        v = nn.emit(permute_dims(v, [0, 2, 1, 3]))

        # Calculate QK
        attn_weights = nn.emit(
            matmul(q, permute_dims(k, [0, 1, 3, 2]))
            / relax.const(
                math.sqrt(self.head_dim),
                q.struct_info.dtype,
            )
        )
        
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            # No prune heads
            position_bias_masked = position_bias
 
        # Apply attention mask
        attn_weights = nn.emit(
            maximum(
                attn_weights,
                relax.const(
                    tvm.tir.min_value(attn_weights.struct_info.dtype).value,
                    attn_weights.struct_info.dtype,
                ),
            )
        )
        attn_weights = nn.emit(minimum(attn_weights, attention_mask))
        # Calculate Softmax(QK)
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != q.struct_info.dtype:
            attn_weights = astype(attn_weights, q.struct_info.dtype)
        # Calculate Softmax(QK)V
        attn_output = nn.emit(matmul(attn_weights, v))
        # Apply output projection
        attn_output = self.dense(
            reshape(
                permute_dims(attn_output, [0, 2, 1, 3]),
                (batch_size, seq_len, self.hidden_size),
            )
        )
        return attn_output, past_key_value

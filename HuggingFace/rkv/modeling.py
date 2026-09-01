import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from .batching import (
    build_additive_attention_mask,
    compress_by_length_group,
    padding_present,
)
from .compression import (
    R1KV,
    SnapKV,
    StreamingLLM,
    H2O,
    AnalysisKV,
    CovarianceMerge,
    RKVMerge,
    RKVMergeAnchor,
)
from .query_moments import (
    assert_rope_composes,
    beta_from_half_life,
    build_future_rope_operator,
    read_future_moments,
    update_query_moments,
)

KV_COMPRESSION_MAP = {
    "rkv": R1KV,
    "snapkv": SnapKV,
    "streamingllm": StreamingLLM,
    "h2o": H2O,
    "analysiskv": AnalysisKV,
    "covariance_merge": CovarianceMerge,
    "rkv_merge": RKVMerge,
    "rkv_merge_anchor": RKVMergeAnchor,
}

logger = logging.get_logger(__name__)



def covariance_merge_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    transformers' `eager_attention_forward` plus one addition: the per-KV-slot additive logit bias
    `module.merge_beta` (bsz, num_kv_heads, kv_len), folded in before the softmax. That bias is what
    makes attending to a merged centroid approximate attending to all of its members.

    `merge_beta` is always sized to whatever this forward actually attends over, which is the
    *uncompressed* cache on a step where compression fires -- see `_compression_step`.
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    beta = getattr(module, "merge_beta", None)
    if beta is not None:
        # (bsz, num_kv_heads, kv_len) -> (bsz, num_heads, 1, kv_len): broadcast over the query heads
        # of a GQA group (they share the slot, hence the bias) and over every query position here
        bias = beta.repeat_interleave(module.num_key_value_groups, dim=1).unsqueeze(2)
        attn_weights = attn_weights + bias.to(attn_weights.dtype)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _merge_metadata_for_step(self, past_key_value, key_states):
    """`(beta, n)` for CovarianceMerge, extended with a fresh `(0, 1)` slot per token added this
    forward -- i.e. mirroring `cat(cached, new)`, so the pair stays index-aligned with the
    post-`update()` cache. Must be called BEFORE that update.

    Stored on `past_key_value.beta_cache` / `.n_cache`, dicts keyed by layer_idx exactly like the
    pre-existing `query_cache`, so they reset per `generate()` call for free.
    """
    if not hasattr(past_key_value, "beta_cache"):
        past_key_value.beta_cache = {}
        past_key_value.n_cache = {}
    beta_new = key_states.new_zeros(*key_states.shape[:3], dtype=torch.float32)
    n_new = key_states.new_ones(*key_states.shape[:3], dtype=torch.float32)
    if self.layer_idx not in past_key_value.beta_cache:
        return beta_new, n_new
    return (
        torch.cat([past_key_value.beta_cache[self.layer_idx], beta_new], dim=2),
        torch.cat([past_key_value.n_cache[self.layer_idx], n_new], dim=2),
    )


def _store_merge_metadata(self, past_key_value, metadata, is_merge):
    if is_merge:
        past_key_value.beta_cache[self.layer_idx] = metadata[0]
        past_key_value.n_cache[self.layer_idx] = metadata[1]


def _update_merge_moments(self, past_key_value, prerope_query):
    """Fold this forward's pre-RoPE queries into the layer's future-query EMA. Runs every step;
    CovarianceMerge needs no recent-query window as a result."""
    valid = _valid_or_full(
        getattr(past_key_value, "step_valid_lengths", None), prerope_query
    )
    update_query_moments(
        past_key_value,
        self.layer_idx,
        prerope_query,
        self.config.num_key_value_heads,
        beta_from_half_life(getattr(self.config, "ema_half_life", 64)),
        valid,
        is_prefill=self.config.compression is None,
    )


def _valid_or_full(valid_lengths, tensor):
    """Fall back to 'every row entirely real' when no validity bookkeeping exists."""
    if valid_lengths is not None:
        return valid_lengths
    return torch.full(
        (tensor.shape[0],), tensor.shape[2], dtype=torch.long, device=tensor.device
    )


def _compression_step(self, past_key_value, key_states, value_states, cached_queries, cache_kwargs, attention_mask, q_len):
    """
    Unified compression step for all phases (prefill, decode) and all compression methods.
    3-way:
        config.compression is None -> prefill
        config.compression is True -> decode and compress
        config.compression is False -> decode and no compress
    Implements group-wise compression to support batched decoding.

    Returns `(key_states, value_states, attention_mask)`. The mask is replaced by one derived
    from our own validity bookkeeping whenever padding is present, because HuggingFace's mask
    stops describing the cache layout the moment compression shrinks it.
    """
    update_cache = getattr(self.config, "update_kv", True) is True
    is_merge = getattr(self.kv_cluster, "uses_merge_metadata", False)

    # CovarianceMerge carries two kinds of side state through the generic channels in
    # rkv/batching.py: per-slot (beta, n), which must be re-indexed exactly like the keys, and
    # per-(row, KV head) (mu, Sigma), which has no sequence axis. No other kernel is affected.
    extras, extra_pads, head_extras = (), (), ()
    if is_merge:
        extras = _merge_metadata_for_step(self, past_key_value, key_states)
        # padded beta is inert (those slots are masked out anyway); padded mass 1 keeps every padded
        # slot a well-formed singleton rather than a divide-by-zero waiting to happen
        extra_pads = (0.0, 1.0)

    step_valid = getattr(past_key_value, "step_valid_lengths", None)

    def compress(keys, values, valid_lengths):
        # Read the future-query moments here rather than above: this runs only on steps that
        # actually compress, so the (bsz, Hkv, D, D) projection is skipped on every other step.
        head_extras = (
            read_future_moments(past_key_value, self.layer_idx, past_key_value.future_rope)
            if is_merge
            else ()
        )
        if not padding_present(valid_lengths, keys.shape[2]):
            outputs = self.kv_cluster.update_kv(
                keys, cached_queries, values, *extras, *head_extras
            )
            return outputs, valid_lengths.new_full((keys.shape[0],), outputs[0].shape[2])
        return compress_by_length_group(
            self.kv_cluster.update_kv,
            valid_lengths,
            keys,
            cached_queries,
            values,
            extras=extras,
            extra_pad_values=extra_pads,
            head_extras=head_extras,
        )

    # Prefill branch
    # we compress if valid length exceeds the budget, but we only update the cache.
    # The attention is computed using the *uncompressed* cache.
    if self.config.compression is None:
        attn_valid = step_valid = _valid_or_full(step_valid, key_states)
        attn_extras = extras
        if update_cache:
            outputs, cache_valid = compress(key_states, value_states, step_valid)
            past_key_value.update(outputs[0], outputs[1], self.layer_idx, cache_kwargs)
            _store_merge_metadata(self, past_key_value, outputs[2:], is_merge)
        else:
            past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            cache_valid = step_valid
            _store_merge_metadata(self, past_key_value, extras, is_merge)

    # Decoding branch, compression fired
    # Similar to prefill above, we only update the cache after compression,
    # but the attention is computed using the *uncompressed* cache.
    elif self.config.compression is True:
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        step_valid = _valid_or_full(step_valid, key_states)
        outputs, compressed_valid = compress(key_states, value_states, step_valid)
        attn_valid = step_valid  # This is for uncompressed tensors (for attention)
        attn_extras = extras
        if update_cache:
            past_key_value.key_cache[self.layer_idx] = outputs[0]
            past_key_value.value_cache[self.layer_idx] = outputs[1]
            cache_valid = compressed_valid
            _store_merge_metadata(self, past_key_value, outputs[2:], is_merge)
        else:
            cache_valid = step_valid
            _store_merge_metadata(self, past_key_value, extras, is_merge)

    # Decoding branch, compression not fired
    else:
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
        cache_valid = attn_valid = _valid_or_full(step_valid, key_states)
        attn_extras = extras
        _store_merge_metadata(self, past_key_value, extras, is_merge)

    # every layer computes the same thing here, so writing it unconditionally is idempotent
    past_key_value.kv_valid_lengths = cache_valid  # this is for caching, so takes effect from next decoding step.

    if is_merge:
        # sized to the tensors attention will actually run over this step, NOT the compressed cache
        self.merge_beta = attn_extras[0]

    if getattr(past_key_value, "has_padding", False):
        # Rebuild the mask for the whole generation, not just while some row still has padding.
        # HuggingFace slices `attention_mask[:, :kv_len]` from the FRONT, so the moment
        # compression shrinks the cache that slice lands on the prompt's leading padding zeros
        # instead of the real window -- wrong even for a row that has since become dense. (An
        # all-ones mask is immune, which is why the unpadded path never needed this.)
        attention_mask = build_additive_attention_mask(
            attn_valid, key_states.shape[2], q_len, key_states.dtype, key_states.device
        )

    return key_states, value_states, attention_mask


def LlamaAttention_init(
    self, config: LlamaConfig, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    if getattr(self.kv_cluster, "uses_merge_metadata", False):
        # The mask FORMAT the upstream model code builds is keyed off `_attn_implementation`
        # (bool for sdpa, additive float for eager), independently of which attention function we
        # install below. Overriding only the function leaves eager reading an sdpa-shaped mask,
        # which silently produces immediate-EOS garbage. Force eager so the two agree.
        self.config._attn_implementation = "eager"
        assert_rope_composes(self.config)
    # =============== New logic end =================

def LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    # kept for CovarianceMerge's future-query moments, which model the pre-RoPE query and
    # apply the rotation for future positions themselves (see rkv/query_moments.py)
    prerope_query = query_states
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        # CovarianceMerge keeps online future-query moments instead of a recent-query window, so the
        # window is not maintained for it at all.
        if getattr(self.kv_cluster, "uses_merge_metadata", False):
            _update_merge_moments(self, past_key_value, prerope_query)
        query_window_needed = getattr(self.kv_cluster, "uses_query_window", True)
        if query_window_needed and not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if query_window_needed and (self.layer_idx not in past_key_value.query_cache):
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        elif query_window_needed:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = (
            past_key_value.query_cache[self.layer_idx]
            if query_window_needed
            else query_states
        )
        key_states, value_states, attention_mask = _compression_step(
            self,
            past_key_value,
            key_states,
            value_states,
            cached_queries,
            cache_kwargs,
            attention_mask,
            query_states.shape[2],
        )
        # =============== decoding-time compression end ===============

    attention_interface: Callable = eager_attention_forward
    if getattr(self.kv_cluster, "uses_merge_metadata", False):
        attention_interface = covariance_merge_eager_attention_forward
    elif self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen2Attention_init(
    self, config: Qwen2Config, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    if getattr(self.kv_cluster, "uses_merge_metadata", False):
        # The mask FORMAT the upstream model code builds is keyed off `_attn_implementation`
        # (bool for sdpa, additive float for eager), independently of which attention function we
        # install below. Overriding only the function leaves eager reading an sdpa-shaped mask,
        # which silently produces immediate-EOS garbage. Force eager so the two agree.
        self.config._attn_implementation = "eager"
        assert_rope_composes(self.config)
    # =============== New logic end =================

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    # kept for CovarianceMerge's future-query moments, which model the pre-RoPE query and
    # apply the rotation for future positions themselves (see rkv/query_moments.py)
    prerope_query = query_states
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        # CovarianceMerge keeps online future-query moments instead of a recent-query window, so the
        # window is not maintained for it at all.
        if getattr(self.kv_cluster, "uses_merge_metadata", False):
            _update_merge_moments(self, past_key_value, prerope_query)
        query_window_needed = getattr(self.kv_cluster, "uses_query_window", True)
        if query_window_needed and not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if query_window_needed and (self.layer_idx not in past_key_value.query_cache):
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        elif query_window_needed:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end ===============

        # =============== decoding-time compression start ===============
        cached_queries = (
            past_key_value.query_cache[self.layer_idx]
            if query_window_needed
            else query_states
        )
        key_states, value_states, attention_mask = _compression_step(
            self,
            past_key_value,
            key_states,
            value_states,
            cached_queries,
            cache_kwargs,
            attention_mask,
            query_states.shape[2],
        )
        # =============== decoding-time compression end ===============

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if getattr(self.kv_cluster, "uses_merge_metadata", False):
        attention_interface = covariance_merge_eager_attention_forward
    elif self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen3Attention_init(
    self, config: Qwen3Config, layer_idx: int, compression_config: dict
):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

        # =============== New logic start ===============
        self.config.update(compression_config)
        self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
            **compression_config["method_config"]
        )
        if getattr(self.kv_cluster, "uses_merge_metadata", False):
            self.config._attn_implementation = "eager"
            assert_rope_composes(self.config)
        # =============== New logic end =================

def Qwen3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    # kept for CovarianceMerge's future-query moments, which model the pre-RoPE query and
    # apply the rotation for future positions themselves (see rkv/query_moments.py)
    prerope_query = query_states
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        # CovarianceMerge keeps online future-query moments instead of a recent-query window, so the
        # window is not maintained for it at all.
        if getattr(self.kv_cluster, "uses_merge_metadata", False):
            _update_merge_moments(self, past_key_value, prerope_query)
        query_window_needed = getattr(self.kv_cluster, "uses_query_window", True)
        if query_window_needed and not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if query_window_needed and (self.layer_idx not in past_key_value.query_cache):
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        elif query_window_needed:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = (
            past_key_value.query_cache[self.layer_idx]
            if query_window_needed
            else query_states
        )
        key_states, value_states, attention_mask = _compression_step(
            self,
            past_key_value,
            key_states,
            value_states,
            cached_queries,
            cache_kwargs,
            attention_mask,
            query_states.shape[2],
        )
        # =============== decoding-time compression end ===============

    attention_interface: Callable = eager_attention_forward
    if getattr(self.kv_cluster, "uses_merge_metadata", False):
        attention_interface = covariance_merge_eager_attention_forward
    elif self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # sample-level statistics
    is_prefill = past_key_values is None or len(past_key_values) == 0
    if is_prefill:
        if self.config.compression_content == "think":
            self.after_think = False
        self.config.compression = None  # None means prefill

    # self.length drives the `step_length` compression trigger
    # (is_newline = self.length % divide_length == 0). It MUST reset at every
    # prefill: left running across samples it keeps accumulating, so the phase
    # of the every-divide_length trigger depends on the total token count of
    # all preceding samples. The trigger frequency stayed correct but which
    # absolute step it fired on became order-dependent, making runs
    # irreproducible under any change to dataset order or fraction.
    #
    # Note that `self.length` starts at the *prompt* length, so the trigger phase is
    # `prompt_len % divide_length`. With a padded batch the prompt length is the *padded*
    # length, which makes a row's compression schedule depend on which other prompts shared
    # its batch. `divide_method="generated_length"` below avoids that entirely.
    if is_prefill or not hasattr(self, "length"):
        self.length = input_ids.shape[1]
        self.generated_length = 0
    else:
        self.length += input_ids.shape[1]
        self.generated_length += input_ids.shape[1]

    # =============== batched-decoding bookkeeping start ===============
    # `step_valid_lengths` tells the layers how much of each row of the (shared, physical) KV
    # tensors is real content this step; it stays constant for the whole forward. The layers
    # write back `kv_valid_lengths` describing the cache they leave behind. See rkv/batching.py.
    if past_key_values is not None:
        batch_size = input_ids.shape[0]
        if is_prefill:
            if attention_mask is not None and attention_mask.dim() == 2:
                step_valid = attention_mask.sum(dim=-1).to(torch.long)
            else:
                step_valid = torch.full(
                    (batch_size,), input_ids.shape[1], dtype=torch.long, device=input_ids.device
                )

            if padding_present(step_valid, input_ids.shape[1]):
                # a padded batch needs an explicit additive mask, which flash-attention has no
                # parameter for at all
                if "flash" in self.config._attn_implementation:
                    logger.warning_once(
                        "Batches of unequal-length prompts require an additive attention mask, which "
                        "flash_attention_2 cannot accept; falling back to sdpa for this run. Pass "
                        "equal-length prompts (or batch_size=1) to keep flash-attention."
                    )
                    self.config._attn_implementation = "sdpa"

            # Latched for the whole sequence, not re-derived per step: once the prompt carried
            # padding, HuggingFace's mask is unusable from the first compression onwards (see
            # _compression_step), even after every row has become dense again.
            past_key_values.has_padding = bool(padding_present(step_valid, input_ids.shape[1]))

            # These don't work for batched decoding
            if batch_size > 1:
                # both of these decide compression for the whole batch from row 0's predicted
                # token, which is silently wrong rather than merely approximate once bsz > 1
                if self.config.divide_method == "newline":
                    raise NotImplementedError(
                        "divide_method='newline' decides the compression step from row 0's predicted "
                        "token only, so it cannot drive a batch. Use divide_method='generated_length' "
                        "(batch-wide and prompt-length independent) or 'step_length', or batch_size=1."
                    )
                if self.config.compression_content == "think":
                    raise NotImplementedError(
                        "compression_content='think' tracks the </think> transition from row 0 only, "
                        "so it cannot drive a batch. Use compression_content='all' or batch_size=1."
                    )
        # decoding branch
        else:
            step_valid = past_key_values.kv_valid_lengths + input_ids.shape[1]
        past_key_values.step_valid_lengths = step_valid
    # =============== batched-decoding bookkeeping end =================

    # =============== CovarianceMerge future-query frame start =========
    # The horizon operator depends only on the RoPE frequencies and each row's current position, so
    # it is identical for all layers -- build it once here rather than 28 times. `rotary_emb` lives
    # on the Model, not on the attention layers, so here is also the only place that *can* build it.
    # Only needed on steps that actually compress; the moments themselves update every step.
    if (
        past_key_values is not None
        and getattr(self.config, "method", None) in ("covariance_merge", "rkv_merge", "rkv_merge_anchor")
        and self.config.compression is not False
    ):
        if position_ids is not None:
            frame_positions = position_ids
        else:
            # cache_position is shared across the batch; broadcast it to one row per sequence
            frame_positions = cache_position.view(1, -1).expand(input_ids.shape[0], -1)
        past_key_values.future_rope = build_future_rope_operator(
            self.model.rotary_emb,
            frame_positions,
            getattr(self.config, "future_horizon", 128),
            getattr(self.config, "future_decay", 1.0),
            # fp32 reference: rotary_emb returns cos/sin in the reference's dtype, and rounding them
            # through bf16 before building the horizon coefficients loses precision for free
            torch.empty(0, dtype=torch.float32, device=input_ids.device),
        )
    # =============== CovarianceMerge future-query frame end ===========

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = self.length % self.config.divide_length == 0
    elif self.config.divide_method == "generated_length":
        # This makes the eviction result agnostic to the batch that potentially contains padding
        is_newline = (
            self.generated_length > 0
            and self.generated_length % self.config.divide_length == 0
        )
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think == True:
        is_newline = False

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        layer.self_attn.config.compression = is_newline
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

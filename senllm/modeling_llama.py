import os
import logging
import math
import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Sequence, Set, TYPE_CHECKING

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.llama.modeling_llama import *
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from attention import get_top_attention_head_positions
from senllm.token_matching import TokenSequenceMatcher

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase




CAL_IDX = int(os.environ.get("CAL_IDX", 0))
BETA = float(os.environ.get("BETA", 0))
COEFF = float(os.environ.get("COEFF", 1.0))

logger = logging.get_logger(__name__)

DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD: float = 0.3
DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD: float = 0.95

_CONFIG_FOR_DOC = "LlamaConfig"




def find_token_indices(input_ids, token=32000):
    # 断言 input_ids 中所有序列都包含元素 32000
    assert (input_ids == token).any(dim=1).all(), f"Not all sequences contain the token {token}"
    
    mask = (input_ids == token)
    mask_float = mask.float()
    first_match_indices = mask_float.argmax(dim=1)
    
    return first_match_indices


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self._zero_special_skip_total: int = 0
        self.zero_special_skip_threshold: float = DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD
        self.zero_special_target_threshold: float = DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        average_last_token_attention: bool = False,
        average_attention_mask: Optional[torch.Tensor] = None,
        average_attention_heads: Optional[Sequence[int]] = None,
        average_attention_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        attention_enhance_gamma: Optional[float] = kwargs.pop("attention_enhance_gamma", None)
        attention_enhance_non_special_mask: Optional[torch.Tensor] = kwargs.pop(
            "attention_enhance_non_special_mask", None
        )
        attention_enhance_mode: str = kwargs.pop("attention_enhance_mode", "scale_max")

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        key_mask_tensor: Optional[torch.Tensor] = None
        override_scores: Optional[torch.Tensor] = None
        apply_additive = False
        if average_attention_state is not None:
            key_mask_tensor = average_attention_state.get("key_mask")
            if key_mask_tensor is not None and key_mask_tensor.device != attn_weights.device:
                key_mask_tensor = key_mask_tensor.to(attn_weights.device)
                average_attention_state["key_mask"] = key_mask_tensor
            cumulative_scores = average_attention_state.get("scores")
            cumulative_weight = average_attention_state.get("weight")
            if cumulative_scores is not None and cumulative_weight is not None:
                if cumulative_scores.device != attn_weights.device or cumulative_scores.dtype != attn_weights.dtype:
                    cumulative_scores = cumulative_scores.to(attn_weights.device, dtype=attn_weights.dtype)
                    average_attention_state["scores"] = cumulative_scores
                if cumulative_weight.device != attn_weights.device or cumulative_weight.dtype != attn_weights.dtype:
                    cumulative_weight = cumulative_weight.to(attn_weights.device, dtype=attn_weights.dtype)
                    average_attention_state["weight"] = cumulative_weight
                weight_clamped = cumulative_weight.clamp_min(1e-6)
                override_scores = (cumulative_scores / weight_clamped).to(attn_weights.dtype)
                apply_additive = True

        layer_contrib: Optional[torch.Tensor] = None
        if average_attention_heads:
            layer_contrib = self._collect_layer_average_scores(
                attn_weights,
                average_attention_mask,
                average_attention_heads,
                key_mask_tensor,
            )
            if layer_contrib is not None:
                layer_contrib = layer_contrib.detach()

        if average_last_token_attention and apply_additive and override_scores is not None:
            attn_weights = self._apply_average_attention_distribution(
                attn_weights,
                average_attention_mask,
                average_attention_heads,
                override_scores=override_scores,
                key_mask=key_mask_tensor,
                additive=True,
            )

        if (
            average_attention_state is not None
            and layer_contrib is not None
            and average_attention_heads
            and len(average_attention_heads) > 0
        ):
            weight_value = float(max(1, len(average_attention_heads)))
            contrib_scaled = layer_contrib.to(attn_weights.dtype) * weight_value
            weight_tensor = attn_weights.new_full((contrib_scaled.size(0), 1), weight_value)
            prev_scores = average_attention_state.get("scores")
            prev_weight = average_attention_state.get("weight")
            if prev_scores is None or prev_weight is None:
                average_attention_state["scores"] = contrib_scaled
                average_attention_state["weight"] = weight_tensor
            else:
                if prev_scores.device != contrib_scaled.device or prev_scores.dtype != contrib_scaled.dtype:
                    prev_scores = prev_scores.to(contrib_scaled.device, dtype=contrib_scaled.dtype)
                    average_attention_state["scores"] = prev_scores
                if prev_weight.device != weight_tensor.device or prev_weight.dtype != weight_tensor.dtype:
                    prev_weight = prev_weight.to(weight_tensor.device, dtype=weight_tensor.dtype)
                    average_attention_state["weight"] = prev_weight
                average_attention_state["scores"] = prev_scores + contrib_scaled
                average_attention_state["weight"] = prev_weight + weight_tensor

        attn_weights = self._apply_attention_enhance_last_token(
            attn_weights,
            average_attention_heads,
            attention_enhance_non_special_mask,
            attention_enhance_gamma,
            attention_enhance_mode,
        )

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @staticmethod
    def _apply_average_attention_distribution(
        attn_weights: torch.Tensor,
        mask: Optional[torch.Tensor],
        head_indices: Optional[Sequence[int]],
        override_scores: Optional[torch.Tensor] = None,          # ← 新增
        key_mask: Optional[torch.Tensor] = None,                  # ← 新增
        additive: bool = False,
    ) -> torch.Tensor:
        """将 mask 指定位置的注意力头分布替换为关键头的平均注意力分布。
        扩展：若提供 override_scores (batch, seq_len)，则直接使用它；否则按 head_indices 计算本层均值。
        若提供 key_mask (batch, seq_len)，则仅保留该列集合上的分布并在该集合内重新归一化。
        当 additive=True 时，将 override_scores 作为加性偏置叠加到所有注意力头后再归一化。
        """
        if attn_weights.ndim != 4:
            return attn_weights
        if (not head_indices) and (override_scores is None):
            return attn_weights

        attn_weights = attn_weights.clone()
        batch_size, _, query_len, key_len = attn_weights.shape

        # 归一化/准备 query 侧行掩码
        if mask is None:
            device = attn_weights.device
            mask = torch.zeros(batch_size, query_len, dtype=torch.bool, device=device)
            mask[:, -1] = True
        else:
            if mask.device != attn_weights.device:
                mask = mask.to(attn_weights.device)
            if mask.shape[0] != batch_size or mask.shape[1] != query_len:
                device = attn_weights.device
                tmp = torch.zeros(batch_size, query_len, dtype=torch.bool, device=device)
                tmp[:, -1] = True
                mask = tmp

        # 归一化/准备 key 侧列掩码（限定平均所用 token 范围）
        if key_mask is not None:
            if key_mask.device != attn_weights.device:
                key_mask = key_mask.to(attn_weights.device)
            if key_mask.shape[0] != batch_size or key_mask.shape[1] != key_len:
                key_mask = None

        eps = 1e-12
        for b in range(batch_size):
            row_mask = mask[b]
            pos = torch.nonzero(row_mask, as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            heads_slice = attn_weights[b]

            # 预取跨层累计分布
            override_row = None
            if override_scores is not None:
                override_row = override_scores[b]
                if override_row.device != heads_slice.device:
                    override_row = override_row.to(heads_slice.device)
                if override_row.dim() != 1 or override_row.numel() != key_len:
                    override_row = None

            for p in pos.tolist():
                if 0 <= p < heads_slice.shape[1]:
                    if override_row is not None:
                        avg_scores = override_row
                    else:
                        key_head_scores = heads_slice[head_indices, p, :]
                        if key_head_scores.numel() == 0:
                            continue
                        avg_scores = key_head_scores.mean(dim=0)

                    # 可选：只在指定 key 列集合内保留并归一化
                    if key_mask is not None:
                        col_mask = key_mask[b]
                        if col_mask.device != avg_scores.device:
                            col_mask = col_mask.to(avg_scores.device)
                        avg_scores = avg_scores * col_mask.to(avg_scores.dtype)
                        if not additive:
                            s = avg_scores.sum()
                            if float(s) > eps:
                                avg_scores = avg_scores / s  # 仅在有效列集合内归一化

                    if float(avg_scores.sum()) > eps:
                        avg_scores = avg_scores / avg_scores.sum()
                    expanded = avg_scores.unsqueeze(0).expand(heads_slice.size(0), -1)
                    # print(f"{expanded}")
                    # print(f"{heads_slice[:, p, :]}")
                    if additive:
                        # print(f"Expanded Sum: {torch.sum(expanded)}")
                        # print(f"Expanded Max: {torch.max(expanded)}")
                        # print(f"heads_slice Sum(before): {torch.sum(heads_slice[:, p, :])}")
                        # print(f"heads_slice Max(before): {torch.max(heads_slice[:, p, :])}")
                        heads_slice[:, p, :] = heads_slice[:, p, :] + 0.3*expanded
                        denom = heads_slice[:, p, :].sum(dim=-1, keepdim=True).clamp_min(eps)
                        heads_slice[:, p, :] = heads_slice[:, p, :] / denom
                        # print(f"heads_slice Sum(after): {torch.sum(heads_slice[:, p, :])}")
                        # print(f"heads_slice Max(after): {torch.max(heads_slice[:, p, :])}")
                    else:
                        heads_slice[:, p, :] = expanded
        return attn_weights

    @staticmethod
    def _collect_layer_average_scores(
        attn_weights: torch.Tensor,
        mask: Optional[torch.Tensor],
        head_indices: Optional[Sequence[int]],
        key_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """计算当前层关键头在查询位置上的平均注意力分布。"""
        if attn_weights.ndim != 4:
            return None
        if not head_indices:
            return None
        batch_size, _, query_len, key_len = attn_weights.shape
        device = attn_weights.device

        if mask is None:
            mask = torch.zeros(batch_size, query_len, dtype=torch.bool, device=device)
            mask[:, -1] = True
        else:
            if mask.device != device:
                mask = mask.to(device)
            if mask.shape[0] != batch_size or mask.shape[1] != query_len:
                tmp = torch.zeros(batch_size, query_len, dtype=torch.bool, device=device)
                tmp[:, -1] = True
                mask = tmp

        key_mask_local = None
        if key_mask is not None:
            key_mask_local = key_mask
            if key_mask_local.device != device:
                key_mask_local = key_mask_local.to(device)
            if key_mask_local.shape[0] != batch_size or key_mask_local.shape[1] != key_len:
                key_mask_local = None

        result = attn_weights.new_zeros(batch_size, key_len)
        has_value = False
        index_list = [int(idx) for idx in head_indices]
        for b in range(batch_size):
            row_mask = mask[b]
            positions = torch.nonzero(row_mask, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            head_tensor = attn_weights[b, index_list][:, positions, :]
            if head_tensor.numel() == 0:
                continue
            mean_scores = head_tensor.mean(dim=0).mean(dim=0)
            if key_mask_local is not None:
                mean_scores = mean_scores * key_mask_local[b].to(mean_scores.dtype)
            result[b] = mean_scores
            has_value = True

        if not has_value:
            return None
        return result

    def _apply_attention_enhance_last_token(
        self,
        attn_weights: torch.Tensor,
        head_indices: Optional[Sequence[int]],
        non_special_mask: Optional[torch.Tensor],
        gamma: Optional[float],
        mode: str,
    ) -> torch.Tensor:
        """当配置了关键注意力头时，调整最后一个 query 的注意力分布。"""
        if attn_weights.ndim != 4:
            return attn_weights
        if not head_indices:
            return attn_weights

        mode_normalized = (mode or "scale_max").strip().lower()
        if mode_normalized not in {"scale_max", "zero_special"}:
            mode_normalized = "scale_max"

        gamma_tensor: Optional[torch.Tensor] = None
        if mode_normalized == "scale_max":
            if gamma is None:
                return attn_weights
            gamma_value = float(gamma)
            if not math.isfinite(gamma_value) or gamma_value <= 0:
                return attn_weights

        batch_size, num_heads, query_len, key_len = attn_weights.shape
        if query_len == 0 or key_len == 0:
            return attn_weights

        means_index = key_len - 5
        if means_index < 0:
            return attn_weights

        target_query = query_len - 1
        device = attn_weights.device
        dtype = attn_weights.dtype

        head_list = sorted({int(h) for h in head_indices if 0 <= int(h) < num_heads})
        if not head_list:
            return attn_weights

        if mode_normalized == "scale_max":
            gamma_tensor = torch.tensor(float(gamma), dtype=dtype, device=device)

        updated_weights = attn_weights
        eps = 1e-12

        non_special_mask_local = None
        if non_special_mask is not None:
            non_special_mask_local = non_special_mask
            if non_special_mask_local.device != device:
                non_special_mask_local = non_special_mask_local.to(device)
            if (
                non_special_mask_local.shape[0] != batch_size
                or non_special_mask_local.shape[1] != key_len
            ):
                non_special_mask_local = None

        num_updates = 0
        skip_enhance_total = 0
        for head_idx in head_list:
            if num_updates >= 2:
                break

            # 维持原先对最后一个 query token 的处理
            base_row = attn_weights[:, head_idx, target_query, :]
            full_max = base_row.max(dim=-1).values
            valid_batches = torch.isfinite(full_max)
            candidate_indices = torch.nonzero(valid_batches, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() > 0:
                if mode_normalized == "zero_special":
                    if means_index <= 0 or means_index >= key_len:
                        continue
                    batch_indices = candidate_indices
                    current_rows = base_row[batch_indices].clone()
                    if current_rows.size(0) == 0:
                        continue
                    keep_mask: Optional[torch.Tensor] = None
                    if non_special_mask_local is not None:
                        keep_mask = non_special_mask_local[batch_indices]
                        if keep_mask.dtype != torch.bool:
                            keep_mask = keep_mask.to(torch.bool)
                        keep_mask = keep_mask.clone()
                    if keep_mask is None:
                        keep_mask = torch.zeros(
                            current_rows.shape, dtype=torch.bool, device=device
                        )
                    keep_mask[:, means_index] = False
                    other_mask = ~keep_mask
                    other_mask[:, means_index] = True
                    removed_mass = (current_rows * other_mask.to(dtype)).sum(dim=-1, keepdim=True)
                    current_rows = current_rows * keep_mask.to(dtype)
                    current_rows[:, means_index] = removed_mass.squeeze(-1)
                    row_sums = current_rows.sum(dim=-1, keepdim=True).clamp_min(eps)
                    current_rows = current_rows / row_sums
                    skip_threshold = getattr(
                        self,
                        "zero_special_skip_threshold",
                        DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD,
                    )
                    target_threshold = getattr(
                        self,
                        "zero_special_target_threshold",
                        DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD,
                    )
                    means_values = current_rows[:, means_index]
                    below_half_mask = means_values < skip_threshold
                    if below_half_mask.any():
                        skip_indices = torch.nonzero(below_half_mask, as_tuple=False).squeeze(-1)
                        skip_enhance_total += int(skip_indices.numel())
                        current_rows[skip_indices] = base_row[batch_indices][skip_indices]

                    enhance_mask = ~below_half_mask
                    if enhance_mask.any():
                        enhance_indices = torch.nonzero(enhance_mask, as_tuple=False).squeeze(-1)
                        enhance_rows = current_rows[enhance_indices]
                        enhance_means = enhance_rows[:, means_index]
                        low_mask = enhance_means < target_threshold
                        boost_iters = 0
                        while low_mask.any() and boost_iters < 8:
                            boost_indices = torch.nonzero(low_mask, as_tuple=False).squeeze(-1)
                            boosted_rows = enhance_rows[boost_indices].clone()
                            boosted_rows[:, means_index] = boosted_rows[:, means_index] * 1.5
                            boosted_rows = boosted_rows / boosted_rows.sum(dim=-1, keepdim=True).clamp_min(eps)
                            enhance_rows[boost_indices] = boosted_rows
                            enhance_means = enhance_rows[:, means_index]
                            low_mask = enhance_means < target_threshold
                            boost_iters += 1
                        current_rows[enhance_indices] = enhance_rows
                    if updated_weights is attn_weights:
                        updated_weights = attn_weights.clone()
                    updated_weights[batch_indices, head_idx, target_query, :] = current_rows
                else:
                    batch_indices = candidate_indices
                    current_rows = base_row[batch_indices].clone()
                    mask_rows: Optional[torch.Tensor] = None
                    if non_special_mask_local is not None:
                        mask_rows = non_special_mask_local[batch_indices]
                        if mask_rows.dtype != torch.bool:
                            mask_rows = mask_rows.to(torch.bool)
                        zero_mask_rows = ~mask_rows
                        current_rows = current_rows.masked_fill(zero_mask_rows, 0.0)
                    else:
                        mask_rows = None
                    current_rows[:, means_index] = 0.0
                    candidate_max = current_rows.max(dim=-1).values
                    fallback_max = full_max[batch_indices].to(dtype)
                    current_max = torch.where(candidate_max > 0, candidate_max, fallback_max)
                    new_values = current_max * gamma_tensor
                    current_rows[:, means_index] = new_values
                    current_rows = torch.clamp_min(current_rows, 0.0)
                    row_sums = current_rows.sum(dim=-1, keepdim=True)
                    zero_mask = row_sums.squeeze(-1) <= 0
                    if zero_mask.any():
                        fallback_rows = base_row[batch_indices][zero_mask]
                        current_rows[zero_mask] = fallback_rows
                        row_sums = current_rows.sum(dim=-1, keepdim=True)
                    current_rows = current_rows / row_sums
                    if updated_weights is attn_weights:
                        updated_weights = attn_weights.clone()
                    updated_weights[batch_indices, head_idx, target_query, :] = current_rows

            num_updates += 1

        if skip_enhance_total:
            self._zero_special_skip_total += int(skip_enhance_total)

        return updated_weights




class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        average_last_token_attention = kwargs.pop("average_last_token_attention", False)
        average_attention_mask = kwargs.pop("average_attention_mask", None)
        average_attention_heads = kwargs.pop("average_attention_heads", None)
        average_attention_state = kwargs.pop("average_attention_state", None)
        attention_enhance_gamma = kwargs.pop("attention_enhance_gamma", None)
        attention_enhance_non_special_mask = kwargs.pop("attention_enhance_non_special_mask", None)
        attention_enhance_mode = kwargs.pop("attention_enhance_mode", "scale_max")
        needs_manual_path = (
            average_last_token_attention
            or average_attention_mask is not None
            or average_attention_heads is not None
            or average_attention_state is not None
            or attention_enhance_gamma is not None
            or attention_enhance_non_special_mask is not None
            or (attention_enhance_mode and attention_enhance_mode != "scale_max")
        )
        if needs_manual_path:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                average_last_token_attention=average_last_token_attention,
                average_attention_mask=average_attention_mask,
                average_attention_heads=average_attention_heads,
                average_attention_state=average_attention_state,
                attention_enhance_gamma=attention_enhance_gamma,
                attention_enhance_non_special_mask=attention_enhance_non_special_mask,
                attention_enhance_mode=attention_enhance_mode,
                **kwargs,
            )
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                average_last_token_attention=average_last_token_attention,
                average_attention_mask=average_attention_mask,
                average_attention_heads=average_attention_heads,
                average_attention_state=average_attention_state,
                attention_enhance_gamma=attention_enhance_gamma,
                attention_enhance_non_special_mask=attention_enhance_non_special_mask,
                attention_enhance_mode=attention_enhance_mode,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value



LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn = LLAMA_ATTENTION_CLASSES["sdpa"](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        pst_token_indices: Optional[torch.Tensor] = None,
        layer_index: Optional[int] = None,
        first_token_indices: Optional[torch.Tensor] = None,
        average_last_token_attention: bool = False,
        average_attention_mask: Optional[torch.Tensor] = None,
        average_attention_heads: Optional[Sequence[int]] = None,
        average_attention_state: Optional[Dict[str, torch.Tensor]] = None,
        attention_enhance_gamma: Optional[float] = None,
        attention_enhance_non_special_mask: Optional[torch.Tensor] = None,
        attention_enhance_mode: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            average_last_token_attention=average_last_token_attention,
            average_attention_mask=average_attention_mask,
            average_attention_heads=average_attention_heads,
            average_attention_state=average_attention_state,
            attention_enhance_gamma=attention_enhance_gamma,
            attention_enhance_non_special_mask=attention_enhance_non_special_mask,
            attention_enhance_mode=attention_enhance_mode,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
        self.plan = 'vanilla'
        self.tp_starting_index = 1
        self.tp_exiting_index = 99
        # 记录注意力增强相关配置与 hook 句柄
        self._attention_enhance_handles = []
        self.attention_enhance_config: Dict = {}
        self._attention_enhance_mask: Optional[torch.Tensor] = None
        self._attention_enhance_log_once = False
        self._attention_enhance_fallback_logged = False
        self._attention_enhance_mask_issue_logged = False
        self._input_device_warning_logged = False
        self._attention_enhance_heads_by_layer: Dict[int, List[int]] = {}
        self._attention_analysis_processed = 0
        self._attention_analysis_records: List[Dict] = []
        self._attention_analysis_current_input_ids: Optional[torch.Tensor] = None
        self._attention_analysis_current_mask: Optional[torch.Tensor] = None
        self._token_matcher: Optional[TokenSequenceMatcher] = None
        self._attention_analysis_text_map: Dict[int, Optional[str]] = {}
        self._attention_analysis_pending_texts: Optional[List[Optional[str]]] = None
        self._attention_analysis_text_matchers: List[Optional[TokenSequenceMatcher]] = []
        self._attention_analysis_tokenizer: Optional["PreTrainedTokenizerBase"] = None
        self.summary_layer_index: Optional[int] = None
        self._summary_hidden_cache: Dict[int, torch.Tensor] = {}
        self._average_last_token_attention: bool = False
        self._average_last_token_start_layer: Optional[int] = None
        self._average_attention_special_token_ids: Optional[Set[int]] = None
        self._attention_enhance_gamma: Optional[float] = None
        self._attention_enhance_override_enabled: bool = True
        self._default_special_token_ids: Set[int] = set()
        self._default_special_token_ids = self._collect_default_special_token_ids()
        self._attention_enhance_mode: str = "scale_max"
        self._zero_special_skip_threshold: float = DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD
        self._zero_special_target_threshold: float = DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_zero_special_skip_total(self) -> int:
        total = 0
        for layer in self.layers:
            attn_module = getattr(layer, "self_attn", None)
            if attn_module is None:
                continue
            total += int(getattr(attn_module, "_zero_special_skip_total", 0))
        return total

    def set_attention_analysis_texts(
        self,
        texts: Optional[List[Optional[str]]],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    ) -> None:
        self._attention_analysis_pending_texts = list(texts) if texts is not None else None
        self._attention_analysis_text_matchers = []
        self._attention_analysis_tokenizer = tokenizer
        if tokenizer is None or not texts:
            return

        matcher_cache: Dict[str, Optional[TokenSequenceMatcher]] = {}
        resolved_matchers: List[Optional[TokenSequenceMatcher]] = []

        for text in texts:
            text_key = text or ""
            if text_key in matcher_cache:
                resolved_matchers.append(matcher_cache[text_key])
                continue
            if not text:
                matcher_cache[text_key] = None
                resolved_matchers.append(None)
                continue

            matcher = TokenSequenceMatcher.from_phrases(
                [text],
                tokenizer,
                include_leading_space_variant=True,
                prefixes=[" ", '"', ' "'],
                suffixes=['"', '" '],
            )
            if matcher.is_empty():
                matcher_cache[text_key] = None
                resolved_matchers.append(None)
            else:
                matcher_cache[text_key] = matcher
                resolved_matchers.append(matcher)

        self._attention_analysis_text_matchers = resolved_matchers

    def _collect_default_special_token_ids(self) -> Set[int]:
        special_ids: Set[int] = set()
        candidate_attrs = (
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
            "mask_token_id",
            "cls_token_id",
            "sep_token_id",
        )
        for attr in candidate_attrs:
            value = getattr(self.config, attr, None)
            if value is not None and isinstance(value, (int, float)):
                int_value = int(value)
                if int_value >= 0:
                    special_ids.add(int_value)
        additional = getattr(self.config, "special_tokens_map_extended", None)
        if isinstance(additional, dict):
            for token_list in additional.values():
                if isinstance(token_list, (list, tuple)):
                    for token_info in token_list:
                        if isinstance(token_info, dict):
                            token_id = token_info.get("id")
                            if token_id is not None:
                                special_ids.add(int(token_id))
        extra_ids = getattr(self.config, "additional_special_tokens_ids", None)
        if extra_ids is not None:
            for token_id in extra_ids:
                if token_id is not None:
                    special_ids.add(int(token_id))
        return special_ids

    def configure_attention_enhance(self, enhance_config: Optional[Dict] = None) -> None:
        """根据配置开启或关闭注意力增强，默认作用于最后一个 token，可针对指定 token 序列。"""
        self._clear_attention_enhance_hooks()
        self.attention_enhance_config = enhance_config or {}
        self._attention_enhance_mask = None
        self._attention_enhance_log_once = False
        self._attention_enhance_fallback_logged = False
        self._attention_enhance_mask_issue_logged = False
        self._attention_enhance_heads_by_layer = {}
        self._attention_analysis_processed = 0
        self._attention_analysis_records = []
        self._attention_analysis_text_map = {}
        self._attention_analysis_pending_texts = None
        self._average_last_token_attention = bool(
            self.attention_enhance_config.get("average_last_token_attention", False)
        )
        self._attention_enhance_override_enabled = bool(
            self.attention_enhance_config.get("enable_attention_override", True)
        )
        mode_value = enhance_config.get("override_mode") if enhance_config else None
        if mode_value is None and enhance_config:
            mode_value = enhance_config.get("mode")
        mode_normalized = "scale_max"
        if isinstance(mode_value, str):
            candidate_mode = mode_value.strip().lower()
            if candidate_mode in {"scale_max", "zero_special"}:
                mode_normalized = candidate_mode
            else:
                logger.warning(
                    "[attention_enhance] override_mode=%r 不受支持，回退为 scale_max。", mode_value
                )
        self._attention_enhance_mode = mode_normalized
        combined_special_ids: Set[int] = set(self._default_special_token_ids)
        special_token_ids = (enhance_config or {}).get("special_token_ids")
        if special_token_ids:
            try:
                provided_ids = {int(tid) for tid in special_token_ids}
                combined_special_ids.update(provided_ids)
            except (TypeError, ValueError):
                logger.warning(
                    "[attention_enhance] special_token_ids=%r 无法解析，将忽略。",
                    special_token_ids,
                )
        self._average_attention_special_token_ids = combined_special_ids if combined_special_ids else None
        start_layer_value = self.attention_enhance_config.get("average_last_token_start_layer")
        self._average_last_token_start_layer = None
        if start_layer_value is not None:
            try:
                self._average_last_token_start_layer = int(start_layer_value)
                self.attention_enhance_config["average_last_token_start_layer"] = self._average_last_token_start_layer
            except (TypeError, ValueError):
                logger.warning(
                    "[attention_enhance] average_last_token_start_layer=%r 无法解析，将忽略。",
                    start_layer_value,
                )
                self.attention_enhance_config.pop("average_last_token_start_layer", None)
        if not enhance_config or not enhance_config.get("enabled", False):
            return

        gamma = float(enhance_config.get("gamma", 1.0))
        # 允许运营直接指定 heads，否则根据评分文件自动选择 top_k
        heads_setting = enhance_config.get("heads")
        top_k = int(enhance_config.get("top_k", 0))
        score_file = enhance_config.get("score_file")
        head_order = str(self.attention_enhance_config.get("head_order", "score") or "score").lower()

        target_token_ids = enhance_config.get("target_token_ids")
        self._token_matcher = None
        self._summary_hidden_cache = {}
        if target_token_ids:
            matcher_candidate = TokenSequenceMatcher(
                target_token_ids if isinstance(target_token_ids, (list, tuple)) else [target_token_ids]
            )
            if matcher_candidate.is_empty():
                logger.warning("[attention_enhance] 提供的 target_token_ids=%s 无法解析，忽略。", target_token_ids)
                self.attention_enhance_config.pop("target_token_ids", None)
            else:
                self._token_matcher = matcher_candidate
                sequences = matcher_candidate.sequences
                self.attention_enhance_config["target_token_ids"] = sequences if len(sequences) > 1 else sequences[0]
        else:
            self.attention_enhance_config.pop("target_token_ids", None)
            self._token_matcher = None

        candidate_positions: List[Tuple[int, int]] = []
        if heads_setting:
            for head in heads_setting:
                if isinstance(head, (list, tuple)) and len(head) >= 2:
                    layer_idx, head_idx = int(head[0]), int(head[1])
                    candidate_positions.append((layer_idx, head_idx))
        else:
            if top_k <= 0:
                return
            try:
                positions = (
                    get_top_attention_head_positions(score_file=score_file, k=top_k)
                    if score_file
                    else get_top_attention_head_positions(k=top_k)
                )
                for position in positions:
                    if len(position) >= 2:
                        candidate_positions.append((int(position[0]), int(position[1])))
            except Exception as exc:
                logger.warning(f"读取注意力评分文件失败，跳过注意力增强: {exc}")
                return

        heads_by_layer: Dict[int, List[int]] = defaultdict(list)
        for position in candidate_positions:
            if len(position) < 2:
                continue
            layer_idx, head_idx = position[0], position[1]
            if not (0 <= layer_idx < len(self.layers)):
                continue
            num_heads = self.layers[layer_idx].self_attn.num_heads
            if not (0 <= head_idx < num_heads):
                continue
            if head_idx not in heads_by_layer[layer_idx]:
                heads_by_layer[layer_idx].append(head_idx)

        if head_order != "score":
            for layer_idx, head_list in heads_by_layer.items():
                head_list.sort()

        skip_raw = enhance_config.get("zero_special_skip_threshold", self._zero_special_skip_threshold)
        target_raw = enhance_config.get("zero_special_target_threshold", self._zero_special_target_threshold)
        try:
            skip_threshold = float(skip_raw)
        except (TypeError, ValueError):
            skip_threshold = float(
                self._zero_special_skip_threshold or DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD
            )
        try:
            target_threshold = float(target_raw)
        except (TypeError, ValueError):
            target_threshold = float(
                self._zero_special_target_threshold or DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD
            )
        self._zero_special_skip_threshold = skip_threshold
        self._zero_special_target_threshold = target_threshold
        self.attention_enhance_config["zero_special_skip_threshold"] = skip_threshold
        self.attention_enhance_config["zero_special_target_threshold"] = target_threshold
        for layer in self.layers:
            attn_module = getattr(layer, "self_attn", None)
            if attn_module is None:
                continue
            if hasattr(attn_module, "zero_special_skip_threshold"):
                attn_module.zero_special_skip_threshold = skip_threshold
            if hasattr(attn_module, "zero_special_target_threshold"):
                attn_module.zero_special_target_threshold = target_threshold

        if not heads_by_layer:
            logger.warning(
                "[attention_enhance] 未找到可用的注意力头配置（heads=%s, top_k=%s），跳过增强。",
                heads_setting,
                top_k,
            )
            return

        logger.info(
            "[attention_enhance] 已启用：mode=%s, gamma=%.3f, heads_by_layer=%s, target_token_ids=%s, score_file=%s, scope=last_token, skip_thr=%.3f, boost_thr=%.3f",
            self._attention_enhance_mode,
            gamma,
            {layer: heads for layer, heads in heads_by_layer.items()},
            self.attention_enhance_config.get("target_token_ids"),
            score_file,
            skip_threshold,
            target_threshold,
        )

        self._attention_enhance_gamma = gamma
        for layer_idx, head_indices in heads_by_layer.items():
            if not head_indices:
                continue
            self._attention_enhance_heads_by_layer[layer_idx] = head_indices

    def _clear_attention_enhance_hooks(self) -> None:
        """移除当前已注册的注意力增强 hook。"""
        for handle in self._attention_enhance_handles:
            handle.remove()
        self._attention_enhance_handles = []
        self._attention_enhance_mask = None
        self._attention_enhance_heads_by_layer = {}
        self._attention_analysis_processed = 0
        self._attention_analysis_records = []
        self._token_matcher = None
        self._attention_analysis_text_map = {}
        self._attention_analysis_pending_texts = None
        self._attention_analysis_text_matchers = []
        self._attention_analysis_tokenizer = None
        self._summary_hidden_cache = {}
        self._average_last_token_attention = False
        self._average_last_token_start_layer = None
        self._average_attention_special_token_ids = None
        self._attention_enhance_gamma = None
        self._attention_enhance_override_enabled = True
        self._attention_enhance_mode = "scale_max"
        self._zero_special_skip_threshold = DEFAULT_ZERO_SPECIAL_SKIP_THRESHOLD
        self._zero_special_target_threshold = DEFAULT_ZERO_SPECIAL_TARGET_THRESHOLD
        for layer in getattr(self, "layers", []):
            attn_module = getattr(layer, "self_attn", None)
            if attn_module is None:
                continue
            if hasattr(attn_module, "_zero_special_skip_total"):
                attn_module._zero_special_skip_total = 0
            if hasattr(attn_module, "zero_special_skip_threshold"):
                attn_module.zero_special_skip_threshold = self._zero_special_skip_threshold
            if hasattr(attn_module, "zero_special_target_threshold"):
                attn_module.zero_special_target_threshold = self._zero_special_target_threshold

    def _compute_attention_enhance_mask(self, input_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """根据配置生成需要增强的 token mask。"""
        if input_ids is None or not self.attention_enhance_config.get("enabled", False):
            return None
        if self._token_matcher and not self._token_matcher.is_empty():
            mask, matches = self._token_matcher.build_mask(input_ids)
            if mask.any():
                sample_positions: List[List[int]] = []
                for batch_idx, match_list in enumerate(matches):
                    for start, end in match_list:
                        sample_positions.append([batch_idx, start, end])
                        if len(sample_positions) >= 5:
                            break
                    if len(sample_positions) >= 5:
                        break
                # if sample_positions:
                    # logger.debug(
                    #     "[attention_enhance] 目标 tokens 匹配成功，示例坐标=%s",
                    #     sample_positions,
                    # )
                return mask.to(input_ids.device)
            if not self._attention_enhance_fallback_logged:
                logger.info(
                    "未在输入序列中找到注意力增强目标 tokens=%s，退化为放大最后一个 token。",
                    self._token_matcher.sequences,
                )
                self._attention_enhance_fallback_logged = True
        else:
            if not self._attention_enhance_fallback_logged:
                logger.debug("[attention_enhance] 未配置目标 tokens，默认增强最后一个 token。")
                self._attention_enhance_fallback_logged = True

        fallback_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        fallback_mask[:, -1] = True
        return fallback_mask

    def _build_input_text_token_mask(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """根据原始输入文本构建 mask，用于识别需聚合的非提示 tokens。"""
        if input_ids is None:
            return None
        if not self._attention_analysis_text_matchers:
            return None

        batch_size, seq_len = input_ids.shape
        if batch_size == 0 or seq_len == 0:
            return None

        device = input_ids.device
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        input_ids_cpu = input_ids.detach().cpu()
        matched = False
        tokenizer = self._attention_analysis_tokenizer

        for idx in range(batch_size):
            if idx >= len(self._attention_analysis_text_matchers):
                break
            matcher = self._attention_analysis_text_matchers[idx]
            if matcher is None or matcher.is_empty():
                continue
            token_row = input_ids_cpu[idx].tolist()
            spans = matcher.find_matches_in_ids(token_row)
            if not spans:
                continue
            for start, end in spans:
                start_i = max(0, int(start))
                end_i = min(seq_len, int(end))
                if tokenizer is not None:
                    while start_i < end_i:
                        piece = tokenizer.decode(
                            [token_row[start_i]],
                            clean_up_tokenization_spaces=False,
                        ).strip()
                        if piece in {"", '"', "'", "“", "”"}:
                            start_i += 1
                            continue
                        break
                    while end_i > start_i:
                        piece = tokenizer.decode(
                            [token_row[end_i - 1]],
                            clean_up_tokenization_spaces=False,
                        ).strip()
                        if piece in {"", '"', "'", "“", "”"}:
                            end_i -= 1
                            continue
                        break
                if end_i <= start_i:
                    continue
                mask[idx, start_i:end_i] = True
                matched = True

        if not matched:
            return None
        return mask

    def _build_average_attention_key_mask(self, input_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """构造用于平均注意力的 key 端掩码，过滤特殊 token。"""
        if input_ids is None:
            return None
        if not self._average_last_token_attention:
            return None
        if not self._average_attention_special_token_ids:
            return None
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for token_id in self._average_attention_special_token_ids:
            mask &= input_ids != int(token_id)
        return mask

    def _record_attention_analysis(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        batch_map: Dict[int, int],
        summary_hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """记录注意力分析结果，用于定位关注的 token。"""
        if layer_idx not in self._attention_enhance_heads_by_layer:
            return
        heads = self._attention_enhance_heads_by_layer.get(layer_idx, [])
        if not heads:
            return
        if self._attention_analysis_current_input_ids is None or self._attention_analysis_current_mask is None:
            return

        masked_token_positions = self._attention_analysis_current_mask  # shape [B, T] or None
        input_ids_cpu = self._attention_analysis_current_input_ids
        analysis_limit = int(self.attention_enhance_config.get("analysis_samples", 0) or 0)

        for batch_idx, global_sample_idx in batch_map.items():
            if batch_idx >= attn_weights.size(0):
                continue
            token_mask_row = masked_token_positions[batch_idx] if masked_token_positions is not None else None
            if token_mask_row is None:
                continue
            query_positions = torch.nonzero(token_mask_row, as_tuple=False).flatten()
            if query_positions.numel() == 0:
                continue
            head_tensor = attn_weights[batch_idx, heads][:, query_positions, :]
            query_score_tensor = head_tensor.mean(dim=0)  # [num_queries, seq_len]
            combined_scores = query_score_tensor.sum(dim=0)  # shape [seq_len]
            top_k = min(3, combined_scores.size(0))
            top_scores, top_indices = torch.topk(combined_scores, k=top_k)
            token_ids_row = input_ids_cpu[batch_idx].tolist()
            summary_entry = self._summary_hidden_cache.get(int(global_sample_idx))
            summary_hidden = None
            summary_batch_idx = int(batch_idx)
            if summary_entry is not None:
                summary_hidden, summary_batch_idx = summary_entry
            record = {
                "sample_index": int(global_sample_idx),
                "layer": int(layer_idx),
                "heads": [int(h) for h in heads],
                "query_positions": query_positions.tolist(),
                "top_key_indices": top_indices.tolist(),
                "top_scores": [float(score) for score in top_scores],
                "token_ids": token_ids_row,
                "full_scores": combined_scores.detach().cpu().tolist(),
                "input_text": self._attention_analysis_text_map.get(int(global_sample_idx)),
                "query_scores": query_score_tensor.detach().cpu().tolist(),
                "head_count": int(max(1, len(heads))),
                "batch_idx": summary_batch_idx,
            }
            if summary_hidden is not None:
                record["summary_hidden"] = summary_hidden.clone()
            record["visualize"] = bool(analysis_limit <= 0 or global_sample_idx < analysis_limit)
            self._attention_analysis_records.append(record)

    def pop_attention_analysis_records(self) -> List[Dict]:
        """获取并清空当前记录的注意力分析结果。"""
        records = self._attention_analysis_records.copy()
        analysis_limit = int(self.attention_enhance_config.get("analysis_samples", 0) or 0)
        summary_cache = self._summary_hidden_cache
        assigned = set()
        for record in records:
            sample_idx = int(record.get("sample_index", -1))
            if sample_idx in summary_cache and "summary_hidden" not in record:
                summary_hidden, batch_idx = summary_cache[sample_idx]
                record["summary_hidden"] = summary_hidden.clone()
                record.setdefault("batch_idx", int(batch_idx))
                assigned.add(sample_idx)
        for sample_idx, (summary_hidden, batch_idx) in summary_cache.items():
            if sample_idx in assigned:
                continue
            records.append(
                {
                    "sample_index": int(sample_idx),
                    "layer": int(self.summary_layer_index) if self.summary_layer_index is not None else -1,
                    "heads": [],
                    "query_positions": [],
                    "top_key_indices": [],
                    "top_scores": [],
                    "token_ids": [],
                    "full_scores": [],
                    "input_text": self._attention_analysis_text_map.get(int(sample_idx)),
                    "query_scores": [],
                    "head_count": 1,
                    "batch_idx": int(batch_idx),
                    "summary_hidden": summary_hidden.clone(),
                    "visualize": bool(analysis_limit <= 0 or sample_idx < analysis_limit),
                }
            )
        self._summary_hidden_cache = {}
        self._attention_analysis_records = []
        return records

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        
        if self.plan == "tp":
            pst_token_indices = find_token_indices(input_ids, token=32000)
            first_token_indices = find_token_indices(input_ids, token=1)
        
        if inputs_embeds is None:
            if input_ids is not None:
                embed_device = self.embed_tokens.weight.device
                if input_ids.device != embed_device:
                    if not self._input_device_warning_logged:
                        logger.warning(
                            "[attention_enhance] input_ids.device=%s 与 embedding.device=%s 不一致，自动迁移后再计算。",
                            input_ids.device,
                            embed_device,
                        )
                        self._input_device_warning_logged = True
                    input_ids = input_ids.to(embed_device)
                else:
                    self._input_device_warning_logged = False
            inputs_embeds = self.embed_tokens(input_ids)

        attention_enhance_non_special_mask: Optional[torch.Tensor] = None
        if (
            input_ids is not None
            and self._attention_enhance_heads_by_layer
            and self.attention_enhance_config.get("enabled", False)
        ):
            device = input_ids.device
            text_token_mask: Optional[torch.Tensor] = None
            if self._attention_enhance_mode in {"zero_special", "scale_max"}:
                text_token_mask = self._build_input_text_token_mask(input_ids)
                if text_token_mask is not None and text_token_mask.device != device:
                    text_token_mask = text_token_mask.to(device)
            if text_token_mask is not None:
                attention_enhance_non_special_mask = text_token_mask
                self._attention_enhance_mask_issue_logged = False
            else:
                attention_enhance_non_special_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
                if self._average_attention_special_token_ids:
                    special_ids = torch.tensor(
                        sorted(self._average_attention_special_token_ids),
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    if special_ids.numel() > 0:
                        matches = (input_ids.unsqueeze(-1) == special_ids.view(1, 1, -1)).any(dim=-1)
                        attention_enhance_non_special_mask = ~matches
                if (
                    self._attention_enhance_mode in {"zero_special", "scale_max"}
                    and not self._attention_enhance_mask_issue_logged
                ):
                    logger.info(
                        "[attention_enhance] 未能定位输入文本 tokens，将回退为仅过滤特殊符号。",
                    )
                    self._attention_enhance_mask_issue_logged = True

        self._attention_enhance_mask = self._compute_attention_enhance_mask(input_ids)
        analysis_limit = int(self.attention_enhance_config.get("analysis_samples", 0) or 0)
        bsz = inputs_embeds.size(0)
        batch_analysis_map: Dict[int, int] = {}
        pending_texts = self._attention_analysis_pending_texts or []
        analysis_enabled = bool(self.attention_enhance_config.get("enabled", False))
        if analysis_enabled and input_ids is not None:
            start_index = self._attention_analysis_processed
            for batch_idx in range(bsz):
                global_idx = start_index + batch_idx
                batch_analysis_map[batch_idx] = global_idx
                text_value = pending_texts[batch_idx] if batch_idx < len(pending_texts) else None
                self._attention_analysis_text_map[global_idx] = text_value
            if batch_analysis_map:
                self._attention_analysis_current_input_ids = input_ids.detach().cpu()
                self._attention_analysis_current_mask = (
                    self._attention_enhance_mask.detach().cpu() if self._attention_enhance_mask is not None else None
                )
            else:
                self._attention_analysis_current_input_ids = None
                self._attention_analysis_current_mask = None
            self._attention_analysis_processed = start_index + bsz
        else:
            self._attention_analysis_current_input_ids = None
            self._attention_analysis_current_mask = None
            if analysis_enabled:
                self._attention_analysis_processed += bsz
        self._attention_analysis_pending_texts = None
        self._attention_analysis_text_matchers = []
        self._attention_analysis_tokenizer = None
        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        

        hidden_states = inputs_embeds
        if (
            attention_enhance_non_special_mask is not None
            and attention_enhance_non_special_mask.device != hidden_states.device
        ):
            attention_enhance_non_special_mask = attention_enhance_non_special_mask.to(hidden_states.device)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        avg_start_idx: Optional[int] = None
        avg_end_idx: Optional[int] = None
        if self._average_last_token_attention and self.summary_layer_index is not None:
            total_layers = len(self.layers)
            summary_idx = int(self.summary_layer_index)
            if total_layers > 0:
                summary_idx = max(0, min(summary_idx, total_layers - 1))
                start_idx_cfg = self._average_last_token_start_layer
                if start_idx_cfg is None:
                    start_idx = summary_idx
                else:
                    start_idx = int(start_idx_cfg)
                    if start_idx < 0:
                        start_idx += total_layers
                    start_idx = max(0, min(start_idx, total_layers - 1))
                avg_start_idx = min(start_idx, summary_idx)
                avg_end_idx = max(start_idx, summary_idx)
        average_attention_state: Optional[Dict[str, torch.Tensor]] = None
        if avg_start_idx is not None and avg_end_idx is not None:
            key_mask_tensor = self._build_average_attention_key_mask(input_ids)
            if key_mask_tensor is not None and key_mask_tensor.device != hidden_states.device:
                key_mask_tensor = key_mask_tensor.to(hidden_states.device)
            average_attention_state = {
                "scores": None,
                "weight": None,
                "key_mask": key_mask_tensor,
            }
        attention_override_mode = (self._attention_enhance_mode or "scale_max").lower()
        if attention_override_mode not in {"scale_max", "zero_special"}:
            attention_override_mode = "scale_max"
        if not self._attention_enhance_override_enabled:
            attention_override_mode = "scale_max"
        attention_enhance_gamma_value: Optional[float] = None
        if (
            attention_override_mode == "scale_max"
            and self._attention_enhance_override_enabled
            and self._attention_enhance_heads_by_layer
            and self._attention_enhance_gamma is not None
        ):
            attention_enhance_gamma_value = float(self._attention_enhance_gamma)

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            mask_tensor = self._attention_enhance_mask
            if mask_tensor is not None:
                if mask_tensor.shape[1] != seq_len:
                    mask_tensor = None
                else:
                    if mask_tensor.device != hidden_states.device:
                        mask_tensor = mask_tensor.to(hidden_states.device)
            if mask_tensor is None:
                mask_tensor = torch.zeros(bsz, seq_len, dtype=torch.bool, device=hidden_states.device)
                mask_tensor[:, -1] = True

            head_indices = self._attention_enhance_heads_by_layer.get(index)
            # 跳过层  14
            if index == 14:
                head_indices = None
            average_attention_heads: Optional[Sequence[int]] = list(head_indices) if head_indices else None
            average_attention_mask = mask_tensor
            average_last_token_attention = (
                avg_start_idx is not None
                and avg_end_idx is not None
                and avg_start_idx <= index <= avg_end_idx
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    None,
                    None,
                    None,
                    average_last_token_attention,
                    average_attention_mask,
                    average_attention_heads,
                    average_attention_state,
                    attention_enhance_gamma_value,
                    attention_enhance_non_special_mask,
                    attention_override_mode,
                )
            else:
                if self.plan == "vanilla":
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        average_last_token_attention=average_last_token_attention,
                        average_attention_mask=average_attention_mask,
                        average_attention_heads=average_attention_heads,
                        average_attention_state=average_attention_state,
                        attention_enhance_gamma=attention_enhance_gamma_value,
                        attention_enhance_non_special_mask=attention_enhance_non_special_mask,
                        attention_enhance_mode=attention_override_mode,
                    )
                elif self.plan == "tp":
                    layer_index = self.tp_starting_index
                    exiting_index = self.tp_exiting_index
                    assert layer_index < exiting_index
                    if index < layer_index:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            pst_token_indices=pst_token_indices,
                            layer_index=index,
                            first_token_indices=first_token_indices,
                            average_last_token_attention=average_last_token_attention,
                            average_attention_mask=average_attention_mask,
                            average_attention_heads=average_attention_heads,
                            average_attention_state=average_attention_state,
                            attention_enhance_gamma=attention_enhance_gamma_value,
                            attention_enhance_non_special_mask=attention_enhance_non_special_mask,
                            attention_enhance_mode=attention_override_mode,
                        )
                    elif index >= layer_index and index < exiting_index:
                        B = hidden_states.shape[0]
                        previous_sentence_embeddings = hidden_states[:, -1, :].clone()
                        hidden_states[torch.arange(B), pst_token_indices, :] = previous_sentence_embeddings

                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            pst_token_indices=pst_token_indices,
                            layer_index=index,
                            first_token_indices=first_token_indices,
                            average_last_token_attention=average_last_token_attention,
                            average_attention_mask=average_attention_mask,
                            average_attention_heads=average_attention_heads,
                            average_attention_state=average_attention_state,
                            attention_enhance_gamma=attention_enhance_gamma_value,
                            attention_enhance_non_special_mask=attention_enhance_non_special_mask,
                            attention_enhance_mode=attention_override_mode,
                        )
                    elif index >= exiting_index:
                        layer_outputs = decoder_layer(
                            hidden_states,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            past_key_value=past_key_values,
                            output_attentions=output_attentions,
                            use_cache=use_cache,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            pst_token_indices=pst_token_indices,
                            layer_index=index,
                            first_token_indices=first_token_indices,
                            average_last_token_attention=average_last_token_attention,
                            average_attention_mask=average_attention_mask,
                            average_attention_heads=average_attention_heads,
                            average_attention_state=average_attention_state,
                            attention_enhance_gamma=attention_enhance_gamma_value,
                            attention_enhance_non_special_mask=attention_enhance_non_special_mask,
                            attention_enhance_mode=attention_override_mode,
                        )
                    else:
                        raise ValueError("layer index error!")
                
                else:
                    raise ValueError(f"The {self.plan} plan have not yet been implemented!")

            hidden_states = layer_outputs[0]

            if self.summary_layer_index is not None and index == self.summary_layer_index and batch_analysis_map:
                cached_hidden = hidden_states.detach().cpu()
                for batch_idx, global_idx in batch_analysis_map.items():
                    if 0 <= batch_idx < cached_hidden.size(0):
                        self._summary_hidden_cache[int(global_idx)] = (cached_hidden[batch_idx], int(batch_idx))

            if output_attentions and self._attention_enhance_heads_by_layer:
                attn_weights = layer_outputs[1]
                if attn_weights is not None and batch_analysis_map:
                    summary_hidden_states = None
                    self._record_attention_analysis(
                        layer_idx=index,
                        attn_weights=attn_weights,
                        batch_map=batch_analysis_map,
                        summary_hidden_states=None,
                    )

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

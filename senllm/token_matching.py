from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


class TokenSequenceMatcher:
    """帮助在 token 序列中定位指定短语的工具类。"""

    def __init__(self, target_sequences: Sequence[Sequence[int]]):
        normalized = []
        for sequence in target_sequences:
            if not sequence:
                continue
            sanitized = []
            for token_id in sequence:
                try:
                    sanitized.append(int(token_id))
                except (TypeError, ValueError):
                    continue
            if sanitized:
                normalized.append(tuple(sanitized))

        # 去重并按长度排序，方便匹配
        unique_sequences = sorted(set(normalized), key=lambda seq: (len(seq), seq))
        self._target_sequences: List[Tuple[int, ...]] = list(unique_sequences)

    @property
    def sequences(self) -> List[List[int]]:
        """返回内部保存的目标 token 序列（list of list 格式）。"""
        return [list(seq) for seq in self._target_sequences]

    def is_empty(self) -> bool:
        return len(self._target_sequences) == 0

    @staticmethod
    def _extract_input_ids(encoded) -> List[int]:
        if encoded is None:
            return []
        if isinstance(encoded, dict):
            return list(encoded.get("input_ids", []))
        if hasattr(encoded, "input_ids"):
            return list(encoded.input_ids)
        return list(encoded)

    @classmethod
    def from_phrases(
        cls,
        phrases: Iterable[str],
        tokenizer,
        include_leading_space_variant: bool = True,
        prefixes: Iterable[str] | None = None,
        suffixes: Iterable[str] | None = None,
    ) -> "TokenSequenceMatcher":
        """根据原始短语和 tokenizer 生成匹配器。"""
        sequences: List[List[int]] = []
        prefixes = list(prefixes or [])
        suffixes = list(suffixes or [])

        def add_sequence(text: str) -> None:
            encoded = tokenizer(text, add_special_tokens=False)
            token_ids = cls._extract_input_ids(encoded)
            if token_ids:
                sequences.append(token_ids)

        for phrase in phrases:
            if not phrase:
                continue
            add_sequence(phrase)
            if include_leading_space_variant and phrase and not phrase.startswith(" "):
                add_sequence(" " + phrase)

            for prefix in prefixes:
                add_sequence(f"{prefix}{phrase}")
            for suffix in suffixes:
                add_sequence(f"{phrase}{suffix}")
            for prefix in prefixes:
                for suffix in suffixes:
                    add_sequence(f"{prefix}{phrase}{suffix}")
        return cls(sequences)

    def find_matches_in_ids(self, token_ids: Sequence[int]) -> List[Tuple[int, int]]:
        """返回所有匹配 (start, end) 区间，end 为开区间。"""
        matches: List[Tuple[int, int]] = []
        if not self._target_sequences:
            return matches

        token_ids_list = [int(t) for t in token_ids]
        length = len(token_ids_list)

        for sequence in self._target_sequences:
            target_len = len(sequence)
            if target_len == 0 or target_len > length:
                continue
            last_start = length - target_len + 1
            for start in range(last_start):
                if token_ids_list[start : start + target_len] == list(sequence):
                    matches.append((start, start + target_len))
        return matches

    def build_mask(
        self,
        token_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]]]:
        """针对批量 token ids 构建匹配 mask，并返回各样本的匹配区间。"""
        if token_batch is None:
            raise ValueError("token_batch cannot be None")
        if token_batch.ndim != 2:
            raise ValueError("token_batch must be a 2D tensor shaped [batch, seq_len]")

        device = token_batch.device
        mask = torch.zeros(token_batch.shape, dtype=torch.bool, device=device)
        matches_per_sample: List[List[Tuple[int, int]]] = []

        if self.is_empty():
            return mask, matches_per_sample

        token_list_batch = token_batch.tolist()
        for batch_idx, token_ids in enumerate(token_list_batch):
            match_list = self.find_matches_in_ids(token_ids)
            matches_per_sample.append(match_list)
            for start, end in match_list:
                mask[batch_idx, start:end] = True

        return mask, matches_per_sample

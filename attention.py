import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


DEFAULT_SCORE_FILE = Path("./head_score/llama-2-7b-80k.json")


def get_top_attention_head_positions(
    score_file: Union[str, Path] = DEFAULT_SCORE_FILE,
    k: int = 25,
) -> List[Tuple[int, ...]]:
    """Return the positions of the top-k attention heads ordered by score."""
    score_path = Path(score_file)
    with score_path.open("r", encoding="utf-8") as file:
        head_scores = json.loads(file.readline())

    averaged_scores = [
        (tuple(int(index) for index in key.split("-")), float(np.mean(scores)))
        for key, scores in head_scores.items()
    ]
    averaged_scores.sort(key=lambda item: item[1], reverse=True)
    print(averaged_scores[:k])
    return [position for position, _ in averaged_scores[:k]]


def get_layer_sorted_head_indices(
    score_file: Union[str, Path] = DEFAULT_SCORE_FILE,
) -> Dict[int, List[int]]:
    """Return per-layer head indices sorted by descending score."""
    score_path = Path(score_file)
    with score_path.open("r", encoding="utf-8") as file:
        head_scores = json.loads(file.readline())

    layer_entries: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for key, scores in head_scores.items():
        parts = key.split("-")
        if len(parts) < 2:
            continue
        layer_idx = int(parts[0])
        head_idx = int(parts[1])
        mean_score = float(np.mean(scores))
        layer_entries[layer_idx].append((head_idx, mean_score))

    sorted_map: Dict[int, List[int]] = {}
    for layer_idx, head_list in layer_entries.items():
        head_list.sort(key=lambda item: item[1], reverse=True)
        sorted_map[layer_idx] = [head for head, _ in head_list]
    return sorted_map


if __name__ == "__main__":
    print(get_top_attention_head_positions())
    top_positions = get_top_attention_head_positions()
    head_counts = Counter(position[0] for position in top_positions if position)
    print("Per-layer head counts among top positions:")
    layer_num = 0
    for layer in sorted(head_counts):
        layer_num += 1
        print(f"Layer {layer}: {head_counts[layer]}")
    print(f"layer_num: {layer_num}")

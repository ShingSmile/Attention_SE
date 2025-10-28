import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np


DEFAULT_SCORE_FILE = Path("./head_score/llama-2-7b-80k.json")


# top 25
# Layer 6: [9, 16, 30]
# Layer 7: [12, 4]
# Layer 8: [26]
# Layer 11: [15, 2]
# Layer 13: [23]
# Layer 14: [18]
# Layer 15: [14]
# Layer 16: [19, 1]
# Layer 17: [22, 0, 18]
# Layer 18: [30]
# Layer 19: [15]
# Layer 20: [30]
# Layer 21: [30, 1, 16]
# Layer 24: [29, 30]
# Layer 29: [19]

# top 50
# Layer 6: [9, 16, 30]
# Layer 7: [12, 4]
# Layer 8: [26, 22]
# Layer 10: [18]
# Layer 11: [15, 2]
# Layer 13: [23]
# Layer 14: [18, 29, 7, 24, 15, 3]
# Layer 15: [14]
# Layer 16: [19, 1, 30]
# Layer 17: [22, 0, 18, 16, 13]
# Layer 18: [30, 10]
# Layer 19: [15, 10, 14]
# Layer 20: [30, 29, 0, 3, 1]
# Layer 21: [30, 1, 16, 4]
# Layer 22: [22, 8, 19, 30, 27]
# Layer 24: [29, 30, 3]
# Layer 26: [28]
# Layer 29: [19]

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
    result = averaged_scores[:k]
    return [position for position, _ in result]
    # return result


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
    print(top_positions)
    layer_data = defaultdict(list)

    for position in top_positions:
        if not position:  # 过滤掉可能的 None 或空元素
            continue

        (layer, index), score = position

        # 存储 (score, index) 元组，方便后续按 score 排序
        layer_data[layer].append((score, index))

    # 2. 按层的顺序 (sorted keys) 遍历并打印
    print("Top attention head indices per layer (sorted by score):")

    # 3. 按 layer 的 key (0, 1, 2...) 排序
    for layer in sorted(layer_data.keys()):
        # 4. 获取该层的数据列表，例如 [(0.9, 5), (0.95, 3), (0.7, 1)]
        items_list = layer_data[layer]

        # 5. 按 score (元组的第一个元素) 降序排序
        # key=lambda x: x[0] 表示按元组的第0个元素(score)排序
        items_list.sort(key=lambda x: x[0], reverse=True)

        # 6. 提取排序后的 index (元组的第二个元素)
        sorted_indices = [index for score, index in items_list]

        # 7. 打印结果
        print(f"Layer {layer}: {sorted_indices}")

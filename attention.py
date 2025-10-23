import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


DEFAULT_SCORE_FILE = Path("./head_score/llama-2-7b-80k.json")


def get_top_attention_head_positions(
    score_file: Union[str, Path] = DEFAULT_SCORE_FILE,
    k: int = 20,
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


if __name__ == "__main__":
    print(get_top_attention_head_positions())

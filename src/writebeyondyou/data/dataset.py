import random
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class HandwritingDataset(Dataset):
    def __init__(self, directory: str = "data") -> None:
        root = Path(directory)

        self.handwriting = load_dataset(
            "parquet",
            data_files=str(root / "handwriting.parquet"),
            split="train",
        )

        self.reference = load_dataset(
            "parquet",
            data_files=str(root / "reference.parquet"),
            split="train",
        )

        self.reference_index = {
            char: i
            for i, char in enumerate(self.reference["character_id"])
        }

        self.writer_ranges = {}

        start = 0

        for writer_id, group in groupby(self.handwriting["writer_id"]):
            count = sum(1 for _ in group)

            if writer_id in self.writer_ranges:
                raise ValueError(f"Non-contiguous writer: {writer_id}")

            self.writer_ranges[writer_id] = (start, start + count)
            start += count


    def __len__(self) -> int:
        return len(self.handwriting)

    def __getitem__(self, index: int) -> dict:
        target = self.handwriting[index]

        start, end = self.writer_ranges[target["writer_id"]]

        style_indices = random.sample(
            range(start, end),
            k=10,
        )

        style_rows = self.handwriting[style_indices]["strokes"]

        styles = [
            torch.from_numpy(
                np.asarray(strokes, dtype=np.float32).reshape(-1, 3)
            )
            for strokes in style_rows
        ]

        char = target["character_id"]

        content = self.reference[self.reference_index[char]]

        return {
            "writer_id": target["writer_id"],
            "character_id": char,
            "content": torch.from_numpy(
                np.asarray(content["strokes"], dtype=np.float32).reshape(-1, 3)
            ),
            "target": torch.from_numpy(
                np.asarray(target["strokes"], dtype=np.float32).reshape(-1, 3)
            ),
            "style": styles,
        }

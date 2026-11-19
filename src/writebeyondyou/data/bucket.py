import torch

BUCKETS = (64, 128, 192, 256, 384, 576, 768)

def bucket_size(length: int) -> int:
    for size in BUCKETS:
        if length <= size:
            return size

    raise ValueError(f"Sequence too long: {length}")

def collate_strokes(
    sequences: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [seq.size(0) for seq in sequences],
        dtype=torch.long,
    )

    size = bucket_size(int(lengths.max()))

    batch = torch.zeros(len(sequences), size, 3)

    for i, seq in enumerate(sequences):
        batch[i, :seq.size(0)] = seq

    return batch, lengths

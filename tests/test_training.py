from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from writebeyondyou.data.dataset import HandwritingDataset
from writebeyondyou.training.train import train


class TinyDataset(Dataset):
    def __init__(self, size: int = 4) -> None:
        self.samples = [
            {
                "content": torch.tensor([[float(index + 1), 0.0, 1.0]]),
                "target": torch.tensor([[float(index + 2), 0.0, 1.0]]),
                "style": [torch.tensor([[0.0, float(index + 1), 1.0]])],
            }
            for index in range(size)
        ]
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        self.accessed.append(index)
        return self.samples[index]


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        content = batch["content"][:, 0, 0].mean()
        target = batch["target"][:, 0, 0].mean()
        prediction = self.weight * content
        xy_loss = (prediction - target).square()
        pen_loss = (self.weight - 1.0).square()
        return xy_loss + pen_loss, xy_loss, pen_loss


def test_train_updates_and_resumes_from_saved_batch_cursor(tmp_path: Path) -> None:
    dataset = TinyDataset()
    model = TinyModel()
    initial_weight = model.weight.detach().clone()
    checkpoint_dir = tmp_path / "checkpoints"

    first = train(
        checkpoint_dir=checkpoint_dir,
        max_steps=1,
        epochs=None,
        batch_size=1,
        lr=0.01,
        device="cpu",
        checkpoint_every=1,
        seed=23,
        dataset=dataset,
        model=model,
    )

    checkpoint_path = checkpoint_dir / "latest.pt"
    assert first == {
        "global_step": 1,
        "epoch": 0,
        "batch_in_epoch": 1,
        "checkpoint": checkpoint_path,
    }
    assert not torch.equal(model.weight.detach(), initial_weight)
    assert checkpoint_path.is_file()
    assert not list(checkpoint_dir.glob("*.tmp"))

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["global_step"] == 1
    assert saved["epoch"] == 0
    assert saved["batch_in_epoch"] == 1
    assert torch.equal(saved["model"]["weight"], model.weight.detach())

    # The next seeded permutation item proves resume skips the completed batch.
    permutation = torch.randperm(4, generator=torch.Generator().manual_seed(23)).tolist()
    dataset.accessed.clear()
    resumed_model = TinyModel()
    resumed = train(
        checkpoint_dir=checkpoint_dir,
        max_steps=2,
        epochs=None,
        batch_size=1,
        lr=0.01,
        device="cpu",
        resume="latest",
        checkpoint_every=1,
        seed=23,
        dataset=dataset,
        model=resumed_model,
    )

    assert dataset.accessed == [permutation[1]]
    assert resumed == {
        "global_step": 2,
        "epoch": 0,
        "batch_in_epoch": 2,
        "checkpoint": checkpoint_path,
    }
    final = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert final["global_step"] == 2
    assert final["epoch"] == 0
    assert final["batch_in_epoch"] == 2
    assert not torch.equal(final["model"]["weight"], saved["model"]["weight"])


def test_style_sampling_is_deterministic_per_epoch_and_sample() -> None:
    dataset = HandwritingDataset.__new__(HandwritingDataset)
    dataset.set_epoch(epoch=3, seed=17)
    first = dataset._style_indices(index=42, start=100, end=130)
    assert dataset._style_indices(index=42, start=100, end=130) == first

    dataset.set_epoch(epoch=4, seed=17)
    assert dataset._style_indices(index=42, start=100, end=130) != first

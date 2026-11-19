from pathlib import Path

import pytest
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
        self.seen_samples: list[int] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        content = batch["content"][:, 0, 0].mean()
        self.seen_samples.extend(batch["content"][:, 0, 0].int().tolist())
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


@pytest.mark.parametrize("resumed_batch_size", [16, 32, 64])
def test_resume_can_resize_batch_without_skipping_or_repeating_samples(
    tmp_path: Path, resumed_batch_size: int
) -> None:
    dataset = TinyDataset(size=160)
    checkpoint_dir = tmp_path / f"checkpoints-{resumed_batch_size}"
    initial_model = TinyModel()
    initial = train(
        checkpoint_dir=checkpoint_dir,
        max_steps=2,
        batch_size=4,
        device="cpu",
        checkpoint_every=1,
        seed=31,
        dataset=dataset,
        model=initial_model,
    )

    permutation = torch.randperm(160, generator=torch.Generator().manual_seed(31)).tolist()
    assert initial_model.seen_samples == [index + 1 for index in permutation[:8]]
    state = torch.load(initial["checkpoint"], map_location="cpu", weights_only=False)
    assert state["samples_in_epoch"] == 8
    assert state["global_step"] == 2

    with pytest.raises(ValueError, match="batch_size does not match"):
        train(
            checkpoint_dir=checkpoint_dir,
            max_steps=3,
            batch_size=resumed_batch_size,
            device="cpu",
            resume="latest",
            seed=31,
            dataset=dataset,
            model=TinyModel(),
        )

    dataset.accessed.clear()
    resumed_model = TinyModel()
    resumed = train(
        checkpoint_dir=checkpoint_dir,
        max_steps=3,
        batch_size=resumed_batch_size,
        device="cpu",
        resume="latest",
        allow_batch_size_change=True,
        checkpoint_every=1,
        seed=31,
        dataset=dataset,
        model=resumed_model,
    )
    expected = permutation[8 : 8 + resumed_batch_size]
    assert resumed_model.seen_samples == [index + 1 for index in expected]
    assert dataset.accessed == expected
    assert resumed["global_step"] == 3
    assert resumed["epoch"] == 0
    assert resumed["batch_in_epoch"] == 3
    assert resumed["checkpoint"] == initial["checkpoint"]
    final_state = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    assert final_state["samples_in_epoch"] == 8 + len(expected)
    assert final_state["batch_size"] == resumed_batch_size


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_after_partial_last_batch_starts_next_epoch(tmp_path: Path, legacy: bool) -> None:
    dataset = TinyDataset(size=5)
    first = train(
        checkpoint_dir=tmp_path, max_steps=2, batch_size=4, dataset=dataset, model=TinyModel()
    )
    checkpoint = first["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # A periodic checkpoint can capture the cursor before the epoch rolls over.
    state.update(epoch=0, batch_in_epoch=2, samples_in_epoch=5)
    if legacy:
        del state["samples_in_epoch"]
    torch.save(state, checkpoint)

    dataset.accessed.clear()
    result = train(
        checkpoint_dir=tmp_path,
        max_steps=3,
        batch_size=2,
        resume="latest",
        allow_batch_size_change=True,
        dataset=dataset,
        model=TinyModel(),
    )
    next_epoch = torch.randperm(5, generator=torch.Generator().manual_seed(1)).tolist()
    assert dataset.accessed == next_epoch[:2]
    assert result["epoch"] == 1
    assert result["batch_in_epoch"] == 1

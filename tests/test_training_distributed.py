"""Small CPU/Gloo regressions for the torchrun training path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from writebeyondyou.training.train import train


class _VariableStrokeDataset(Dataset):
    """A tiny variable length corpus whose content identifies each sample."""

    def __init__(self, size: int = 7) -> None:
        self.samples = []
        for index in range(size):
            length = index % 3 + 1
            target = torch.zeros(length, 3)
            target[:, 0] = float(index + 1)
            target[:, 1] = float((index % 4) + 1)
            target[:, 2] = float(index % 2)
            self.samples.append(
                {
                    "content": torch.tensor([[float(index + 1), 0.0, 1.0]]),
                    "target": target,
                    "style": [torch.tensor([[0.0, 1.0, 1.0]])],
                }
            )
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        self.accessed.append(index)
        return self.samples[index]


class _TokenWeightedModel(nn.Module):
    """Return local token means, matching the production model loss contract."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.3))
        self.seen: list[int] = []

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        self.seen.extend(batch["content"][:, 0, 0].to(torch.int64).tolist())
        lengths = batch["target_lengths"]
        steps = torch.arange(batch["target"].size(1), device=lengths.device)[None, :]
        xy_mask = (steps < lengths[:, None]).to(batch["target"].dtype)
        pen_mask = (steps <= lengths[:, None]).to(batch["target"].dtype)
        target = batch["target"]
        xy_error = (self.weight * target[..., 0] - target[..., 1]).square()
        pen_error = (self.weight - target[..., 2]).square()
        xy_loss = (xy_error * xy_mask).sum() / xy_mask.sum().clamp_min(1)
        pen_loss = (pen_error * pen_mask).sum() / pen_mask.sum().clamp_min(1)
        return xy_loss + 2.0 * pen_loss, xy_loss, pen_loss


def _distributed_train_worker(rank: int, rendezvous: str, output_dir: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
    )
    output = Path(output_dir)
    checkpoint_dir = output / "checkpoints"
    # Use file rendezvous so each phase has an isolated process group and no
    # fixed TCP port is needed. Recreate the group to exercise a real restart.
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous}-first", rank=rank, world_size=2
    )
    first_dataset = _VariableStrokeDataset()
    first_model = _TokenWeightedModel()
    try:
        first_result = train(
            checkpoint_dir=checkpoint_dir,
            max_steps=1,
            batch_size=3,
            lr=0.01,
            device="cpu",
            workers=0,
            checkpoint_every=1,
            log_every=100,
            seed=41,
            dataset=first_dataset,
            model=first_model,
            distributed_backend="gloo",
        )
        first_seen = first_model.seen.copy()
        if rank == 0:
            import shutil

            shutil.copyfile(checkpoint_dir / "latest.pt", output / "first-checkpoint.pt")
    finally:
        torch.distributed.destroy_process_group()

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous}-resume", rank=rank, world_size=2
    )
    resumed_dataset = _VariableStrokeDataset()
    resumed_model = _TokenWeightedModel()
    try:
        resumed_result = train(
            checkpoint_dir=checkpoint_dir,
            max_steps=3,
            batch_size=3,
            lr=0.01,
            device="cpu",
            workers=0,
            resume="latest",
            checkpoint_every=1,
            log_every=100,
            seed=41,
            dataset=resumed_dataset,
            model=resumed_model,
            distributed_backend="gloo",
        )
        torch.save(
            {
                "first_result": first_result,
                "first_seen": first_seen,
                "resumed_result": resumed_result,
                "resumed_seen": resumed_model.seen,
                "resumed_accessed": resumed_dataset.accessed,
            },
            output / f"rank-{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed is unavailable")
def test_gloo_training_covers_tail_resumes_cursor_and_matches_single_process(
    tmp_path: Path,
) -> None:
    import torch.multiprocessing as mp

    distributed_dir = tmp_path / "distributed"
    distributed_dir.mkdir()
    rendezvous = str(tmp_path / "gloo-rendezvous")
    mp.spawn(
        _distributed_train_worker,
        args=(rendezvous, str(distributed_dir)),
        nprocs=2,
        join=True,
    )

    states = [
        torch.load(distributed_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(2)
    ]
    checkpoint_path = distributed_dir / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    first_checkpoint = torch.load(
        distributed_dir / "first-checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert states[0]["first_result"] == states[1]["first_result"]
    assert states[0]["resumed_result"] == states[1]["resumed_result"]
    assert states[0]["first_result"]["global_step"] == 1
    assert states[0]["resumed_result"]["global_step"] == 3
    assert first_checkpoint["global_step"] == 1
    assert first_checkpoint["samples_in_epoch"] == 3
    assert "weight" in first_checkpoint["model"]
    assert checkpoint["samples_in_epoch"] == 0
    assert checkpoint["batch_in_epoch"] == 0
    assert checkpoint["epoch"] == 1

    # The seeded global order is consumed once. The final one-sample chunk
    # makes rank 1 perform a zero-weight dummy forward so both ranks stay in
    # lockstep; remove that known dummy when checking real sample coverage.
    order = torch.randperm(7, generator=torch.Generator().manual_seed(41)).tolist()
    expected_by_rank: list[list[int]] = [[], []]
    dummy_by_rank: list[list[int]] = [[], []]
    for offset in range(0, len(order), 3):
        chunk = order[offset : offset + 3]
        for rank in range(2):
            local = chunk[rank::2]
            if local:
                expected_by_rank[rank].extend(index + 1 for index in local)
            else:
                dummy_by_rank[rank].append(chunk[0] + 1)
    for rank in range(2):
        assert states[rank]["first_seen"] + states[rank]["resumed_seen"] == (
            expected_by_rank[rank] + dummy_by_rank[rank]
        )
        resumed_expected = [
            sample
            for sample in expected_by_rank[rank] + dummy_by_rank[rank]
            if sample not in states[rank]["first_seen"]
        ]
        assert states[rank]["resumed_seen"] == resumed_expected
        assert states[rank]["resumed_accessed"] == [sample - 1 for sample in resumed_expected]
    seen_real = [sample for local in expected_by_rank for sample in local]
    assert sorted(seen_real) == list(range(1, 8))
    assert len(seen_real) == len(set(seen_real))

    reference_dataset = _VariableStrokeDataset()
    reference_model = _TokenWeightedModel()
    reference = train(
        checkpoint_dir=tmp_path / "reference",
        max_steps=3,
        batch_size=3,
        lr=0.01,
        device="cpu",
        workers=0,
        checkpoint_every=1,
        log_every=100,
        seed=41,
        dataset=reference_dataset,
        model=reference_model,
    )
    assert reference["global_step"] == states[0]["resumed_result"]["global_step"]
    assert torch.allclose(
        checkpoint["model"]["weight"], reference_model.weight.detach(), atol=1e-7, rtol=1e-6
    )

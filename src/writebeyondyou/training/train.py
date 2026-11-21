"""Training entry point for WriteBeyondYou."""

from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from writebeyondyou.data.collate import collate_handwriting
from writebeyondyou.data.dataset import HandwritingDataset
from writebeyondyou.models.model import HandwritingModel


def _atomic_save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint beside its destination, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the checkpoint-stable model behind an optional DDP wrapper."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _synchronize(device: torch.device) -> None:
    """Wait for an asynchronous accelerator before logging or checkpointing."""
    if device.type == "tpu":
        torch.accelerator.synchronize(device)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        # TPU use is explicit because torch_tpu runtime setup is deployment-specific.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "tpu":
        try:
            import torch_tpu  # noqa: F401  # registers the TPU device backend
        except ImportError as error:
            raise RuntimeError("--device tpu requires the torch_tpu package") from error
        return torch.device("tpu")
    return torch.device(requested)


class _EpochBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches without materializing millions of lists."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        epoch: int,
        start_sample: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        generator = torch.Generator().manual_seed(seed + epoch)
        self.order = torch.randperm(dataset_size, generator=generator)
        self.batch_size = batch_size
        self.start_sample = start_sample
        self.rank = rank
        self.world_size = world_size
        self.global_counts: list[int] = []
        self.local_counts: list[int] = []

    def __iter__(self):
        start = self.start_sample
        self.global_counts.clear()
        self.local_counts.clear()
        for offset in range(start, self.order.numel(), self.batch_size):
            indices = self.order[offset : offset + self.batch_size].tolist()
            local = indices[self.rank :: self.world_size]
            self.global_counts.append(len(indices))
            self.local_counts.append(len(local))
            yield local or [indices[0]]

    def __len__(self) -> int:
        remaining = max(self.order.numel() - self.start_sample, 0)
        return (remaining + self.batch_size - 1) // self.batch_size


def train(
    *,
    data: str | Path = "data",
    checkpoint_dir: str | Path = Path("checkpoints"),
    checkpoint: str | Path | None = None,
    max_steps: int | None = None,
    epochs: int | None = None,
    batch_size: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    workers: int = 0,
    resume: str | Path | None = None,
    allow_batch_size_change: bool = False,
    checkpoint_every: int = 100,
    log_every: int = 10,
    seed: int = 0,
    dataset: Dataset | None = None,
    model: torch.nn.Module | None = None,
    distributed_backend: str | None = None,
) -> dict[str, Any]:
    """Train and return final progress; dataset/model injection supports smoke tests."""
    if batch_size < 1 or workers < 0 or checkpoint_every < 1 or log_every < 1:
        raise ValueError(
            "batch_size/checkpoint_every/log_every must be positive and workers non-negative"
        )
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when specified")
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive when specified")
    if max_steps is None and epochs is None:
        raise ValueError("at least one of max_steps or epochs must be specified")
    if max_steps is not None and epochs is not None:
        raise ValueError("specify either max_steps or epochs, not both")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    selected_device = _select_device(device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not dist.is_initialized():
        backend = distributed_backend or (
            "tpu_dist"
            if selected_device.type == "tpu"
            else "gloo"
            if selected_device.type == "cpu"
            else "nccl"
        )
        dist.init_process_group(backend=backend)
    if selected_device.type == "cuda" and world_size > 1:
        torch.cuda.set_device(local_rank)
        selected_device = torch.device("cuda", local_rank)
    dataset = dataset if dataset is not None else HandwritingDataset(str(data))
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty")

    model = model if model is not None else HandwritingModel()
    model = model.to(selected_device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if selected_device.type == "cuda" else None
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else checkpoint_dir / "latest.pt"
    global_step, epoch, batch_in_epoch, samples_in_epoch = 0, 0, 0, 0

    if resume is not None:
        resume_path = checkpoint_path if str(resume) == "latest" else Path(resume)
        state = _load_checkpoint(resume_path)
        _base_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state.get("global_step", state.get("step", 0)))
        saved_batch_size = int(state.get("batch_size", batch_size))
        if "epoch" in state and "batch_in_epoch" in state:
            epoch = int(state["epoch"])
            batch_in_epoch = int(state["batch_in_epoch"])
            legacy_cursor = min(batch_in_epoch * saved_batch_size, len(dataset))
            samples_in_epoch = int(state.get("samples_in_epoch", legacy_cursor))
        else:
            batches_per_epoch = (len(dataset) + saved_batch_size - 1) // saved_batch_size
            epoch, batch_in_epoch = divmod(global_step, batches_per_epoch)
            samples_in_epoch = min(batch_in_epoch * saved_batch_size, len(dataset))
        if state.get("seed", seed) != seed:
            raise ValueError("resume seed does not match checkpoint seed")
        if saved_batch_size != batch_size and not allow_batch_size_change:
            raise ValueError("resume batch_size does not match checkpoint batch_size")
        if not 0 <= samples_in_epoch <= len(dataset):
            raise ValueError("checkpoint samples_in_epoch is outside the dataset")
        if state.get("dataset_size", len(dataset)) != len(dataset):
            raise ValueError("resume dataset size does not match checkpoint dataset size")
        if "rng" in state:
            _restore_rng_state(state["rng"])

    def save_progress() -> None:
        _synchronize(selected_device)
        _atomic_save_checkpoint(
            {
                "model": _to_cpu(_base_model(model).state_dict()),
                "optimizer": _to_cpu(optimizer.state_dict()),
                "global_step": global_step,
                "step": global_step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "samples_in_epoch": samples_in_epoch,
                "seed": seed,
                "batch_size": batch_size,
                "dataset_size": len(dataset),
                "rng": _rng_state(),
            },
            checkpoint_path,
        )

    model.train()
    while (max_steps is None or global_step < max_steps) and (epochs is None or epoch < epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch, seed)
        batch_sampler = _EpochBatchSampler(
            len(dataset), batch_size, seed, epoch, samples_in_epoch, rank, world_size
        )
        if len(batch_sampler) == 0:
            epoch += 1
            batch_in_epoch = 0
            samples_in_epoch = 0
            continue
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            collate_fn=partial(
                collate_handwriting,
                pack_style=selected_device.type == "tpu"
                and isinstance(_base_model(model), HandwritingModel),
            ),
            persistent_workers=False,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        log_started = time.perf_counter()
        last_logged_step = global_step
        for batch_index, batch in enumerate(loader):
            batch = {name: value.to(selected_device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, xy_loss, pen_loss = model(batch)
            reported_loss, reported_xy_loss, reported_pen_loss = loss, xy_loss, pen_loss
            if world_size > 1:
                xy_count = batch["target_lengths"].sum().to(dtype=torch.float32)
                pen_count = (batch["target_lengths"] + 1).sum().to(dtype=torch.float32)
                if batch_sampler.local_counts[batch_index] == 0:
                    xy_count.zero_()
                    pen_count.zero_()
                stats = torch.stack(
                    (
                        xy_loss.detach() * xy_count,
                        pen_loss.detach() * pen_count,
                        xy_count,
                        pen_count,
                    )
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                xy_count, pen_count = stats[2], stats[3]
                loss = world_size * (
                    xy_loss * batch["target_lengths"].sum() / xy_count.clamp_min(1)
                    + 2.0
                    * pen_loss
                    * (batch["target_lengths"] + 1).sum()
                    / pen_count.clamp_min(1)
                )
                if batch_sampler.local_counts[batch_index] == 0:
                    loss = loss * 0.0
                reported_xy_loss = stats[0] / xy_count.clamp_min(1)
                reported_pen_loss = stats[1] / pen_count.clamp_min(1)
                reported_loss = reported_xy_loss + 2.0 * reported_pen_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            global_step += 1
            batch_in_epoch += 1
            samples_in_epoch += batch_sampler.global_counts[batch_index]
            if (
                global_step % log_every == 0
                or samples_in_epoch >= len(dataset)
                or (max_steps is not None and global_step >= max_steps)
            ):
                _synchronize(selected_device)
                elapsed = max(time.perf_counter() - log_started, 1e-9)
                interval_steps = global_step - last_logged_step
                if rank == 0:
                    print(
                        f"Step {global_step}: loss={reported_loss.item():.4f}, "
                        f"xy={reported_xy_loss.item():.4f}, pen={reported_pen_loss.item():.4f}, "
                        f"steps/s={interval_steps / elapsed:.2f}",
                        flush=True,
                    )
                log_started = time.perf_counter()
                last_logged_step = global_step
            if rank == 0 and global_step % checkpoint_every == 0:
                save_progress()
            if max_steps is not None and global_step >= max_steps:
                break

        if samples_in_epoch >= len(dataset):
            epoch += 1
            batch_in_epoch = 0
            samples_in_epoch = 0

    if rank == 0:
        save_progress()
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    if world_size > 1:
        dist.barrier()
    return {
        "global_step": global_step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "checkpoint": checkpoint_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the handwriting model")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="legacy path override")
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--max-steps", "--steps", dest="max_steps", type=int)
    stop_group.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "tpu"), default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument(
        "--allow-batch-size-change",
        action="store_true",
        help="resume with a different batch size from the next unconsumed sample",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distributed-backend", choices=("gloo", "nccl", "tpu_dist"), default=None)
    args = parser.parse_args()
    if args.max_steps is None and args.epochs is None:
        args.max_steps = 10
    try:
        train(
            data=args.data,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint=args.checkpoint,
            max_steps=args.max_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            workers=args.workers,
            resume=args.resume,
            allow_batch_size_change=args.allow_batch_size_change,
            checkpoint_every=args.checkpoint_every,
            log_every=args.log_every,
            seed=args.seed,
            distributed_backend=args.distributed_backend,
        )
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

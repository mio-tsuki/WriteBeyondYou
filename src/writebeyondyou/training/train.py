import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from writebeyondyou.data.collate import collate_handwriting
from writebeyondyou.data.dataset import HandwritingDataset
from writebeyondyou.models.model import HandwritingModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
    )
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")

    # Dataset
    dataset = HandwritingDataset(args.data)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_handwriting,
        prefetch_factor=2 if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
    )

    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty")

    # Model
    device = torch.device(args.device)
    model = HandwritingModel().to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # Training
    global_step = 0

    while global_step < args.steps:
        for batch in loader:
            batch = {
                name: value.to(device)
                for name, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            loss, xy_loss, pen_loss = model(batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

            optimizer.step()
            global_step += 1

            print(
                f"Step {global_step}/{args.steps}: "
                f"loss={loss.item():.4f}, "
                f"xy={xy_loss.item():.4f}, "
                f"pen={pen_loss.item():.4f}",
                flush=True,
            )

            if global_step >= args.steps:
                break

    # Checkpoint
    args.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
        },
        args.checkpoint,
    )

    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

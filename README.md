# WriteBeyondYou

WriteBeyondYou trains a handwriting generator from stroke sequences and writer style examples.

## CPU training

Place `handwriting.parquet` and `reference.parquet` in `data/`, then run:

```bash
uv sync --locked
uv run python -m writebeyondyou.training.train --device cpu --steps 10
```

## Colab TPU training

Select a TPU v5e-1 runtime. Use the PyTorch, `torch_tpu` and JAX packages supplied by
the Colab TPU image. The compiled recurrent kernels were tested with PyTorch
2.14.1, `torch_tpu` 0.2.0 and JAX 0.11.2. TorchTPU runs the kernels on the TPU
and PyTorch manages the gradients, optimizer and checkpoints. CPU training does
not require JAX.

In a Colab notebook cell, mount Drive and copy the dataset to the runtime's local disk:

```python
from pathlib import Path
from shutil import copytree
from subprocess import run

from google.colab import drive

drive.mount("/content/drive")
project = Path("/content/WriteBeyondYou")
if not project.exists():
    run(["git", "clone", "https://github.com/mio-tsuki/WriteBeyondYou.git", project], check=True)
else:
    run(["git", "pull", "--ff-only"], cwd=project, check=True)

drive_root = Path("/content/drive/MyDrive/wby")
copytree(drive_root / "data", project / "data", dirs_exist_ok=True)
```

Run the training entry point and save checkpoints to Drive:

```bash
cd /content/WriteBeyondYou
python -m pip install -e .
python -m writebeyondyou.training.train --device tpu --batch-size 384 \
  --max-steps 200 --checkpoint-every 20 \
  --checkpoint-dir /content/drive/MyDrive/wby/checkpoints/v5e1
```

Batch size 384 was tested on v5e-1 with target sequences padded to 192 steps.
TPU training groups style strokes by length and packs their valid encoder outputs
for attention. Longer target sequences or packed style memories use more device memory.

To continue after an interruption, repeat the command with `--resume` and raise
`--max-steps` to the desired total. Keep the dataset and seed the same. To change the
batch size during resume, pass `--allow-batch-size-change` along with the new
`--batch-size`. Training continues at the next unconsumed sample.
`latest.pt` contains the model, optimizer and training progress. For inference, load its
`model` entry into `HandwritingModel` and call `generate()`.

## Multi-chip TPU training

Use a TorchTPU VM with a compatible PyTorch and JAX installation. With the dataset
in `data/` and the project installed, discover the TPU topology before launching
the trainer:

```bash
eval "$(python -m torch.tpu.distributed | sed 's/^/export /')"
torchrun --nproc_per_node="$WORLD_SIZE" -m writebeyondyou.training.train \
  --device tpu --epochs 2 --batch-size 2048 --checkpoint-dir checkpoints
```

`--batch-size` is the global batch size. On eight chips, a full batch of 2048
contains 256 samples per chip. Rank zero writes `latest.pt`. To resume a
single-chip run, pass its checkpoint path to `--resume` and add
`--allow-batch-size-change` if the global batch size changes.

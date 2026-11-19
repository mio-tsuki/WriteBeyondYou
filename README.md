# WriteBeyondYou

WriteBeyondYou trains a handwriting generator from stroke sequences and writer style examples.

## CPU training

Place `handwriting.parquet` and `reference.parquet` in `data/`, then run:

```bash
uv sync --locked
uv run python -m writebeyondyou.training.train --device cpu --steps 10
```

## Colab TPU training

Select a TPU v5e-1 runtime. Use the matched PyTorch and `torch_tpu` packages supplied by the
Colab TPU image. The entry point also requires `datasets` and `numpy`, which `pip install -e .`
installs if the runtime does not already provide compatible versions.

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
python -m writebeyondyou.training.train --device tpu --batch-size 4 \
  --max-steps 200 --checkpoint-every 20 \
  --checkpoint-dir /content/drive/MyDrive/wby/checkpoints/v5e1
```

To continue after an interruption, repeat the command with `--resume` and raise
`--max-steps` to the desired total. Keep the dataset, batch size and seed the same.
`latest.pt` contains the model, optimizer and training progress. For inference, load its
`model` entry into `HandwritingModel` and call `generate()`.

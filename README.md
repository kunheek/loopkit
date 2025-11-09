# LoopKit

LoopKit is a lightweight experiment-management toolkit for machine-learning projects.
It provides reproducible configuration handling, structured logging, resource
monitoring, git snapshotting, and a CLI for driving sweeps and comparing results.

## Features
- Typed YAML configuration loader with `${env:VAR}` and `${now:%Y%m%d}` interpolation
	and rich `+key=value` command-line overrides.
- Rank-aware `ExperimentLogger` that writes human-readable logs, JSONL events,
	CSV metrics, timing summaries, and best-metric checkpoints.
- Optional resource monitoring (CPU, memory, GPU) and system-info snapshots for
	each run directory.
- Git metadata capture (commit SHA, branch, dirty flag, patch diff) that stays in
	sync with the comparison CLI.
- CLI commands for hyperparameter sweeps, metric visualization, and run
	comparison.
- Optional integrations for Weights & Biases and TensorBoard plus PyTorch helpers
	for checkpoints and distributed training utilities.

## Installation

```bash
pip install loopkit
```

For optional integrations:

```bash
# Tracking backends
pip install "loopkit[tracking]"

# Development extras
pip install "loopkit[dev]"
```

LoopKit requires Python 3.8+.

## Quick Start

```python
from loopkit import Config, ExperimentLogger

# Load configuration from YAML (with interpolation + environment variable support)
cfg = Config.from_file("config.yaml")

logger = ExperimentLogger(run_dir="runs/demo")
logger.info("Training started", config_hash=cfg.hash())

with logger.timer("train_step"):
	...  # training logic

logger.log_metric(step=1, split="train", name="loss", value=0.42)
```

Create a run directory with git and system metadata:

```python
from pathlib import Path
from loopkit.git import save_git_info
from loopkit.monitor import log_system_info

run_dir = Path("runs/demo")
save_git_info(run_dir)
log_system_info(run_dir)
```

## Configuration & Overrides
- YAML files support `${env:VAR,default}` placeholders, `${now:%Y%m%d}` timestamps,
	`${git_sha}` and `${config:other.key}` references.
- Command-line overrides accept both `+key=value` and `+key value` forms; nesting
	via dot notation (`+training.lr=0.01`) and type hints (`+batch_size:int=64`) are
	supported.
- Use `override_config(config, unknown_args)` inside argument parsers to merge the
	leftover `argparse` tokens directly into a config instance.

## CLI Usage

The package installs a `loopkit` CLI with three subcommands.

### Hyperparameter Sweeps

```bash
loopkit sweep "python train.py {config}" \
	--config conf/base.yaml \
	--sweep "training.lr=[0.001,0.01]" "model.hidden_dim=[128,256]"
```

This expands the grid, writes each resolved config to `runs/<sweep_name>/exp_XXX/`,
and executes the template command with `{config}` replaced by the file path. Use
`--dry-run` to preview commands without executing.

### Metric Visualization

```bash
loopkit viz runs/exp1 runs/exp2 --metrics loss accuracy --style seaborn
```

Plots selected metrics across runs or, with `--resources`, renders CPU/GPU/memory
usage from `resource_usage.jsonl`.

### Run Comparison

```bash
loopkit compare runs/exp1 runs/exp2 --metrics loss --export comparison.csv
```

Displays best/final metrics, configuration diffs, and git metadata pulled from the
per-run `git_info.json` snapshot.

## Logging & Monitoring
- `ExperimentLogger` writes colored console logs (rank 0), plain text files, and
	JSONL event streams. Metrics land in `metrics.csv`, with automatic best-metric
	tracking and ETA estimation when `total_steps` is supplied.
- `ResourceMonitor` records GPU/CPU/memory stats at a configurable interval and can
	be used directly or via context manager semantics.
- Set environment variables (`LK_LOG_LEVEL`, `LK_TRACK_METRICS`, `LK_TRACK_GIT`,
	etc.) to toggle global behavior at import time.

## PyTorch Utilities
- `loopkit.torch.checkpoint.CheckpointManager` handles rotating checkpoints and
  best-metric retention.
- Distributed helpers (`setup_ddp`, `main_process_only`, `mp_tqdm`, etc.) make it
  easier to bootstrap multi-process training with sane defaults.
- `DistributedEngine` provides a high-level training wrapper that handles model
  preparation, mixed precision, gradient accumulation, and FSDP/DDP wrapping with
  native PyTorch primitives—no external dependencies required.

### Using DistributedEngine

```python
from loopkit.torch import DistributedEngine

engine = DistributedEngine(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
    use_fsdp=False,  # or True for FSDP instead of DDP
)

model, optimizer, dataloader = engine.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with engine.accumulate(model):
        with engine.autocast():
            loss = model(batch)
        engine.backward(loss)
    
    if engine.sync_gradients:
        engine.clip_grad_norm_(model.parameters(), max_norm=1.0)
        engine.optimizer_step(optimizer)
        optimizer.zero_grad()
```## Contributing

Fork the repository, create a feature branch, and submit a pull request. Run the
formatters and linters before opening the PR:

```bash
ruff format
ruff check
pytest
```

Bug reports and feature suggestions are welcome via GitHub issues.

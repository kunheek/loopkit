import os

__version__ = "0.1.3"

# Import submodules to make them available
from . import config, git, logger, monitor, tracking, utils

# Import optional modules
try:
    from . import torch
except ImportError:
    torch = None  # type: ignore

# Import commonly used classes and functions
from .config import Config, override_config
from .logger import ExperimentLogger
from .utils import get_gpu_devices, set_seed, str2bool

# Export public API
__all__ = [
    "__version__",
    "Config",
    "ExperimentLogger",
    "config",
    "debug",
    "force_single_process",
    "get_gpu_devices",
    "git",
    "log_level",
    "logger",
    "monitor",
    "override_config",
    "save_checkpoints",
    "set_seed",
    "str2bool",
    "timer",
    "torch",
    "track_git",
    "track_metrics",
    "tracking",
    "utils",
    "verbose",
]

# Global configuration flags
# These can be modified at runtime or set via environment variables

# Debug mode - enables debug logging and additional checks
# Set via: loopkit.debug = True or LK_DEBUG=1
debug = str2bool(os.environ.get("LK_DEBUG", "0"))

# Logging level - default log level for ExperimentLogger instances
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Set via: loopkit.log_level = "DEBUG" or LK_LOG_LEVEL=DEBUG
log_level = os.environ.get("LK_LOG_LEVEL", "INFO").upper()

# Profiling - controls whether logger.timer() performs timing measurements
# Disable for production to eliminate profiling overhead
# Set via: loopkit.timer = False or LK_TIMER=0
timer = str2bool(os.environ.get("LK_TIMER", "1"))

# Verbose mode - controls console output verbosity
# Disable for cleaner logs when running many experiments
# Set via: loopkit.verbose = False or LK_VERBOSE=0
verbose = str2bool(os.environ.get("LK_VERBOSE", "1"))

# Checkpoint saving - controls whether to save model checkpoints
# Disable for quick experiments or testing to skip I/O
# Set via: loopkit.save_checkpoints = False or LK_SAVE_CHECKPOINTS=0
save_checkpoints = str2bool(os.environ.get("LK_SAVE_CHECKPOINTS", "1"))

# Metrics tracking - controls whether to log metrics to CSV
# Disable for faster iteration when metrics aren't needed
# Set via: loopkit.track_metrics = False or LK_TRACK_METRICS=0
track_metrics = str2bool(os.environ.get("LK_TRACK_METRICS", "1"))

# Git tracking - controls whether to collect git repository information
# Disable to skip git operations (useful in CI/CD or when git is slow)
# Set via: loopkit.track_git = False or LK_TRACK_GIT=0
track_git = str2bool(os.environ.get("LK_TRACK_GIT", "1"))

# Force single process - force single-process mode in distributed environments
# Useful for debugging distributed code without multiple processes
# Set via: loopkit.force_single_process = True or LK_FORCE_SINGLE=1
force_single_process = str2bool(os.environ.get("LK_FORCE_SINGLE", "0"))

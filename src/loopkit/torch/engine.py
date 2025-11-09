"""High-level distributed training engine with native PyTorch.

This module provides a comprehensive wrapper for distributed training that
simplifies common patterns:

  * Process-group initialization (DDP / FSDP)
  * Mixed precision training with automatic gradient scaling
  * Gradient accumulation with ``no_sync`` context management
  * Simple metric gathering and reduction utilities
  * Automatic cleanup via context manager

**When to use this vs. loopkit.torch.mp:**
  - Use ``DistributedEngine`` for a high-level API that handles model preparation,
    gradient accumulation, and mixed precision automatically
  - Use ``loopkit.torch.mp`` functions (``setup_ddp``, ``barrier``, etc.) for
    fine-grained control when manually setting up distributed training

Example:
    >>> from loopkit import ExperimentLogger
    >>> from loopkit.torch import DistributedEngine
    >>>
    >>> # Context manager automatically calls cleanup on exit
    >>> with DistributedEngine(
    ...     mixed_precision="bf16",
    ...     gradient_accumulation_steps=4
    ... ) as engine:
    ...     logger = ExperimentLogger(run_dir="runs/exp1", rank=engine.rank)
    ...     model, optimizer = engine.prepare(model, optimizer)
    ...
    ...     for step, batch in enumerate(dataloader):
    ...         with engine.accumulate(model):
    ...             with engine.autocast():
    ...                 loss = model(batch)
    ...             engine.backward(loss)
    ...
    ...         if engine.sync_gradients:
    ...             engine.clip_grad_norm_(model.parameters(), max_norm=1.0)
    ...             engine.optimizer_step(optimizer)
    ...             optimizer.zero_grad()
    ...
    ...             # Log metrics on main process
    ...             if engine.is_main_process:
    ...                 logger.log_metric(step, "train", "loss", loss.item())
    >>>
    >>> # Or use manual cleanup
    >>> engine = DistributedEngine()
    >>> try:
    ...     # training code
    ...     pass
    >>> finally:
    ...     engine.cleanup()
"""

from __future__ import annotations

import datetime
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # FSDP is optional but required when use_fsdp=True
    from torch.distributed.fsdp import (  # type: ignore
        BackwardPrefetch,
        CPUOffload,
        FullStateDictConfig,
        MixedPrecision,
        ShardingStrategy,  # type: ignore
        StateDictType,
    )
    from torch.distributed.fsdp import (  # type: ignore
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp.wrap import (  # type: ignore
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
except Exception:  # pragma: no cover - handled gracefully at runtime
    FSDP = None
    MixedPrecision = None
    BackwardPrefetch = None
    CPUOffload = None
    ShardingStrategy = None
    FullStateDictConfig = None
    StateDictType = None
    size_based_auto_wrap_policy = None
    transformer_auto_wrap_policy = None


@dataclass
class EngineState:
    rank: int
    world_size: int
    local_rank: int
    mixed_precision: str
    gradient_accumulation_steps: int
    distributed: bool
    mode: str

    def __str__(self) -> str:
        return (
            "DistributedEngine("  # concise, single-line representation
            f"rank={self.rank}, "
            f"world_size={self.world_size}, "
            f"local_rank={self.local_rank}, "
            f"mixed_precision={self.mixed_precision}, "
            f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
            f"distributed={self.distributed}, "
            f"mode={self.mode})"
        )


class DistributedEngine:
    """High-level distributed training wrapper using native PyTorch primitives.

    Provides automatic model preparation, mixed precision training, gradient
    accumulation, and FSDP/DDP wrapping without external dependencies.
    """

    def __init__(
        self,
        *,
        mixed_precision: str = "fp32",
        gradient_accumulation_steps: int = 1,
        use_fsdp: bool = False,
        fsdp_config: Optional[Dict[str, Any]] = None,
        timeout: Optional[datetime.timedelta] = None,
    ) -> None:
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.mixed_precision = (mixed_precision or "fp32").lower()
        self.use_fsdp = bool(use_fsdp)
        self.fsdp_config = fsdp_config or {}
        self._models: List[torch.nn.Module] = []
        self._optimizers: List[torch.optim.Optimizer] = []
        self._accum_step = 0
        self.sync_gradients = True

        self._init_distributed(timeout)
        self.device = self._resolve_device()
        self.autocast_dtype = self._resolve_autocast_dtype()
        self.scaler = (
            torch.amp.GradScaler(enabled=self.autocast_dtype == torch.float16)
            if self.device.type == "cuda"
            else None
        )

        mode = "fsdp" if self.use_fsdp and self.world_size > 1 else "ddp"
        self.distributed_type = mode
        self.state = EngineState(
            rank=self.rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            distributed=self.world_size > 1,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_distributed(self, timeout: Optional[datetime.timedelta]) -> None:
        penalty = timeout or datetime.timedelta(seconds=5400)
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
            return

        env_rank = int(os.environ.get("RANK", 0))
        env_world = int(os.environ.get("WORLD_SIZE", 1))
        env_local_rank = int(os.environ.get("LOCAL_RANK", env_rank))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if env_world > 1:
            dist.init_process_group(backend=backend, timeout=penalty)
            self.rank = env_rank
            self.world_size = env_world
            self.local_rank = env_local_rank
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def _resolve_autocast_dtype(self) -> Optional[torch.dtype]:
        if self.mixed_precision in {"bf16", "bfloat16"} and self.device.type == "cuda":
            return torch.bfloat16
        if self.mixed_precision in {"fp16", "float16"} and self.device.type == "cuda":
            return torch.float16
        return None

    # ------------------------------------------------------------------
    # Public API (drop-in replacements)
    # ------------------------------------------------------------------
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    def autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    @contextmanager
    def accumulate(self, model: torch.nn.Module):
        if self.gradient_accumulation_steps == 1:
            self.sync_gradients = True
            yield
            return

        next_step = self._accum_step + 1
        self.sync_gradients = (next_step % self.gradient_accumulation_steps) == 0

        if isinstance(model, (DDP, FSDP)) and not self.sync_gradients:
            with model.no_sync():
                yield
        else:
            yield

        self._accum_step = next_step % self.gradient_accumulation_steps

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm_(
        self, parameters: Iterable[torch.nn.Parameter], max_norm: float
    ) -> torch.Tensor:
        if max_norm is None or max_norm <= 0:
            return torch.tensor(0.0, device=self.device)
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return torch.tensor(0.0, device=self.device)
        if self.scaler is not None:
            for optimizer in self._optimizers:
                self.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        return norm

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, device=self.device)
        tensor = tensor.detach()
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if self.world_size == 1:
            return tensor
        outputs = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.stack(outputs)

    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().clone()
        if self.world_size == 1:
            return value
        dist.all_reduce(value, dist.ReduceOp.SUM)
        value /= float(self.world_size)
        return value

    def cleanup(self) -> None:
        """Clean up distributed process group.

        Should be called at the end of training to properly release resources.
        Safe to call even if process group was not initialized.
        """
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def prepare(self, *objects: Any) -> Tuple[Any, ...] | Any:
        prepared: List[Any] = []
        for obj in objects:
            if isinstance(obj, torch.nn.Module):
                prepared.append(self._prepare_model(obj))
            elif isinstance(obj, torch.optim.Optimizer):
                self._optimizers.append(obj)
                prepared.append(obj)
            else:
                prepared.append(obj)
        if len(prepared) == 1:
            return prepared[0]
        return tuple(prepared)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, DDP):
            return model.module
        if FSDP is not None and isinstance(
            model, FSDP
        ):  # pragma: no cover - requires torch>=1.12
            return model.module
        return model

    def get_state_dict(
        self, model: torch.nn.Module
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self.use_fsdp and FSDP is not None and isinstance(model, FSDP):
            config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
                state_dict = model.state_dict()
            return state_dict if self.is_main_process else None
        return self.unwrap_model(model).state_dict()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically cleanup."""
        self.cleanup()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model.to(self.device)
        if self.world_size == 1:
            return model

        if self.use_fsdp:
            if FSDP is None:
                raise RuntimeError(
                    "torch.distributed.fsdp is required but not available"
                )
            fsdp_kwargs = self._build_fsdp_kwargs()
            fsdp_model = FSDP(model, **fsdp_kwargs)
            self._models.append(fsdp_model)
            return fsdp_model

        ddp_model = DDP(
            model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=self.fsdp_config.get(
                "find_unused_parameters", False
            ),
        )
        self._models.append(ddp_model)
        return ddp_model

    def _build_fsdp_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.device.type == "cuda":
            kwargs["device_id"] = self.device

        cfg = self.fsdp_config

        sharding = cfg.get("sharding_strategy", "full").lower()
        if ShardingStrategy is not None:
            if sharding == "hybrid":
                kwargs["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
            elif sharding in {"shard_grad_op", "grad"}:
                kwargs["sharding_strategy"] = ShardingStrategy.SHARD_GRAD_OP
            else:
                kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD

        if MixedPrecision is not None and self.autocast_dtype is not None:
            kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=cfg.get("param_dtype", self.autocast_dtype),
                reduce_dtype=cfg.get("reduce_dtype", self.autocast_dtype),
                buffer_dtype=cfg.get("buffer_dtype", torch.float32),
            )

        if "sync_module_states" in cfg:
            kwargs["sync_module_states"] = bool(cfg["sync_module_states"])
        else:
            kwargs.setdefault("sync_module_states", True)

        if "limit_all_gathers" in cfg:
            kwargs["limit_all_gathers"] = bool(cfg["limit_all_gathers"])
        else:
            kwargs.setdefault("limit_all_gathers", True)

        if "forward_prefetch" in cfg:
            kwargs["forward_prefetch"] = bool(cfg["forward_prefetch"])

        if BackwardPrefetch is not None and "backward_prefetch" in cfg:
            backward_prefetch = str(cfg["backward_prefetch"]).lower()
            if backward_prefetch == "backward_pre":
                kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
            elif backward_prefetch == "backward_post":
                kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_POST

        if "use_orig_params" in cfg:
            kwargs["use_orig_params"] = bool(cfg["use_orig_params"])

        if "cpu_offload" in cfg:
            if CPUOffload is None:
                raise RuntimeError(
                    "torch.distributed.fsdp.CPUOffload is required but unavailable; update PyTorch to enable cpu_offload."
                )
            cpu_offload_cfg = cfg["cpu_offload"]
            if isinstance(cpu_offload_cfg, bool):
                kwargs["cpu_offload"] = CPUOffload(offload_params=cpu_offload_cfg)
            elif isinstance(cpu_offload_cfg, dict):
                kwargs["cpu_offload"] = CPUOffload(**cpu_offload_cfg)
            else:
                raise ValueError(
                    "fsdp_config['cpu_offload'] must be a bool or mapping of CPUOffload keyword arguments."
                )

        if "auto_wrap_policy" in cfg:
            policy_name = str(cfg["auto_wrap_policy"]).lower()
            if (
                policy_name in {"transformer", "transformer_based"}
                and transformer_auto_wrap_policy is not None
            ):
                cls_names = cfg.get("transformer_cls_to_wrap", [])
                if not cls_names:
                    raise ValueError(
                        "fsdp_config['transformer_cls_to_wrap'] must be provided when using the transformer auto wrap policy."
                    )
                transformer_classes = tuple(
                    self._import_object(name) for name in cls_names
                )
                kwargs["auto_wrap_policy"] = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=transformer_classes,
                    **cfg.get("auto_wrap_kwargs", {}),
                )
            elif (
                policy_name in {"size", "size_based"}
                and size_based_auto_wrap_policy is not None
            ):
                min_params = int(cfg.get("min_num_params", 1_000_000))
                kwargs["auto_wrap_policy"] = partial(
                    size_based_auto_wrap_policy,
                    min_num_params=min_params,
                )

        return kwargs

    def _import_object(self, dotted_path: str) -> Any:
        if not dotted_path:
            raise ValueError("Empty import path supplied in fsdp_config.")
        module_path, _, attr = dotted_path.rpartition(".")
        if not module_path:
            raise ValueError(
                f"fsdp_config expects fully qualified class names. Provide module path for '{dotted_path}'."
            )
        module = import_module(module_path)
        try:
            return getattr(module, attr)
        except AttributeError as exc:  # pragma: no cover - configuration error
            raise ValueError(
                f"Cannot import '{attr}' from '{module_path}' while processing fsdp_config."
            ) from exc


__all__ = ["DistributedEngine", "EngineState"]

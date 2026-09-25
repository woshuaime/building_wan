"""Verify that two CUDA devices can synchronize DDP gradients."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import socket
import sys
import traceback

for _name, _value in {
    "PYTHONUNBUFFERED": "1",
    "PYTHONFAULTHANDLER": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "CUDA_MODULE_LOADING": "LAZY",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
}.items():
    os.environ.setdefault(_name, _value)
try:
    faulthandler.enable(all_threads=True)
except (RuntimeError, OSError):
    pass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def resolve_backend(value: str) -> str:
    backend = (value or "auto").strip().lower()
    if backend == "auto":
        return "gloo" if sys.platform == "win32" else "nccl"
    if backend not in {"gloo", "nccl"}:
        raise ValueError(f"Unsupported backend: {value}")
    if backend == "nccl" and sys.platform == "win32":
        raise ValueError("NCCL is not supported on Windows.")
    return backend


def configure_environment(backend: str) -> None:
    for name, value in {
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "TORCH_SHOW_CPP_STACKTRACES": "1",
    }.items():
        os.environ.setdefault(name, value)
    if backend == "nccl":
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    try:
        faulthandler.enable(all_threads=True)
    except (RuntimeError, OSError):
        pass


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _worker(local_rank: int, world_size: int, backend: str, init_method: str):
    configure_environment(backend)
    rank = int(os.environ.get("RANK", local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True
    try:
        model = torch.nn.Linear(4, 1, bias=False, device=device)
        torch.nn.init.constant_(model.weight, 1.0)
        model = DistributedDataParallel(model, device_ids=[local_rank])
        sample = torch.full((2, 4), float(rank + 1), device=device)
        loss = model(sample).square().mean()
        loss.backward()
        checksum = float(model.module.weight.grad.float().sum().item())
        gathered: list[float | None] = [None] * world_size
        dist.all_gather_object(gathered, checksum)
        synchronized = max(gathered) - min(gathered) < 1e-6
        if rank == 0:
            print(
                json.dumps(
                    {
                        "backend": backend,
                        "world_size": world_size,
                        "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
                        "gradient_checksums": gathered,
                        "synchronized": synchronized,
                    },
                    ensure_ascii=False,
                )
            )
        if not synchronized:
            raise RuntimeError("DDP gradients differ across ranks.")
    except BaseException:
        print(
            f"[rank {rank}] DDP smoke test failed:\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    args = parser.parse_args()
    backend = resolve_backend(args.backend)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit("At least two CUDA devices are required.")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _worker(
            int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
            int(os.environ["WORLD_SIZE"]),
            backend,
            "env://",
        )
        return
    if sys.platform != "win32":
        raise SystemExit("Linux requires torchrun for this check.")
    master_port = free_local_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, backend, f"tcp://127.0.0.1:{master_port}"),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

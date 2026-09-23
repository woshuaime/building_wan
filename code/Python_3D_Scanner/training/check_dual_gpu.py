"""Verify that two CUDA devices can synchronize DDP gradients."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _worker(local_rank: int, world_size: int, backend: str, init_method: str):
    rank = local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
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
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="gloo")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit("At least two CUDA devices are required.")
    rendezvous_path = Path(tempfile.gettempdir()) / "python_3d_scanner_ddp_init"
    rendezvous_path.unlink(missing_ok=True)
    init_method = rendezvous_path.as_uri()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, args.backend, init_method),
        nprocs=world_size,
        join=True,
    )
    rendezvous_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

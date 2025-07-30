import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import pandas as pd

MB = 1 << 20
GB = 1 << 30

WORLD_SIZES = [2, 4, 6]
DATA_SIZES = [1 * MB, 16 * MB, 128 * MB, GB]
results = []


def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank: int, world_size: int, data_size: int) -> None:
    setup(rank, world_size)

    data = torch.randint(0, 10, (3,), dtype=torch.int64)

    begin = time.time()
    dist.all_reduce(data)
    end = time.time()

    total_time = end - begin
    dist.all_reduce(total_time)

    if rank == 0:
        avg_time = total_time / world_size
        res = {"world_size": world_size,
               "data_size": data_size, "avg_time": avg_time}
        results.append(res)
        print(res)


def main():
    for world_size in WORLD_SIZES:
        for data_size in DATA_SIZES:
            print(f"Spawning {world_size} processes on data size {
                  data_size}...")
            mp.spawn(fn=distributed_demo, args=(
                world_size,), nprocs=world_size, join=True)

    print(results)


if __name__ == "__main__":
    main()

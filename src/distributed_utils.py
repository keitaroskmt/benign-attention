import torch.distributed as dist


def setup(rank: int, world_size: int) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()

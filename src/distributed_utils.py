import os
import logging
import contextlib

import torch.distributed as dist


def setup(rank: int, world_size: int) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


# https://github.com/huggingface/transformers/blob/004530aa050efcdd489f1ac6809626fa578636ad/src/transformers/training_args.py#L2446
@contextlib.contextmanager
def main_process_first(local=True, desc="work"):
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.

    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    Args:
        local (`bool`, *optional*, defaults to `True`):
            if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
            rank 0 In multi-node environment with a shared filesystem you most likely will want to use
            `local=False` so that only the main process of the first node will do the processing. If however, the
            filesystem is not shared, then the main process of each node will need to do the processing, which is
            the default behavior.
        desc (`str`, *optional*, defaults to `"work"`):
            a work description to be used in debug logs
    """
    logger = logging.getLogger(__name__)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        main_process_desc = "main local process" if local else "main procecs"
        is_main_process = local_rank == 0 if local else rank == 0

        try:
            if not is_main_process:
                # tell all replicas to wait
                logger.debug(
                    f"{rank}: waiting for the {main_process_desc} to perform {desc}"
                )
                dist.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(
                    f"{rank}: {main_process_desc} completed {desc}, releasing all replicas"
                )
                dist.barrier()
    else:
        yield

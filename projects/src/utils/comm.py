# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use detectron2's `launch()` "
    "to start processes and initialize pytorch process group. If you need to start "
    "processes in other ways, please call comm.create_local_process_group("
    "num_workers_per_machine) after calling torch.distributed.init_process_group()."
)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    Detectron2's launch() in engine/launch.py will call this function. If you start
    workers without launch(), you'll have to also call this. Otherwise utilities
    like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically
          the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """
    Returns:
        A torch process group which only includes processes that are on the same
        machine as the current process. This group can be useful for communication
        within a machine, e.g. a per-machine SyncBN.
    """
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return _LOCAL_PROCESS_GROUP


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def reduce_sum(tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def reduce_mean(tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus

def compute_ious(pred, target):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = (area_intersect + 1.0) / (area_union + 1.0)
    gious = ious - (ac_uion - area_union) / ac_uion

    return ious, gious
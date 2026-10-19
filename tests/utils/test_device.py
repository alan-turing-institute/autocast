import pytest
import torch

from autocast.utils.device import pin_local_cuda_device


@pytest.fixture
def fake_cuda(monkeypatch):
    """Four visible GPUs; records every set_device call instead of touching CUDA."""
    calls: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    for name in ("LOCAL_RANK", "SLURM_LOCALID"):
        monkeypatch.delenv(name, raising=False)
    return calls


def test_pin_selects_the_ranks_gpu_from_local_rank(fake_cuda, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    assert pin_local_cuda_device() == 2
    assert fake_cuda == [2]


def test_pin_falls_back_to_the_srun_task_id(fake_cuda, monkeypatch):
    monkeypatch.setenv("SLURM_LOCALID", "3")
    assert pin_local_cuda_device() == 3
    assert fake_cuda == [3]


def test_pin_does_nothing_without_a_local_rank(fake_cuda):
    assert pin_local_cuda_device() is None
    assert fake_cuda == []


def test_pin_does_nothing_without_cuda(fake_cuda, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pin_local_cuda_device() is None
    assert fake_cuda == []


def test_pin_leaves_a_one_gpu_per_task_binding_alone(fake_cuda, monkeypatch):
    # srun binding one GPU per task: task 3 sees a single device, already cuda:0.
    monkeypatch.setenv("SLURM_LOCALID", "3")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert pin_local_cuda_device() is None
    assert fake_cuda == []

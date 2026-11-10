"""Tests for :mod:`src.utils.torch_device`."""

from __future__ import annotations

import pytest

from src.utils import torch_device
from src.utils.torch_device import resolve_device


def test_autodetect_never_picks_mps_unless_asked(monkeypatch) -> None:
    monkeypatch.setattr(torch_device, "device_available", lambda name: name == "mps")
    assert resolve_device(None) == "cpu"
    assert resolve_device(None, prefer_mps=True) == "mps"


def test_autodetect_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch_device, "device_available", lambda name: True)
    assert resolve_device(None) == "cuda"


def test_cpu_is_always_available() -> None:
    assert resolve_device("cpu") == "cpu"


def test_an_unavailable_request_raises(monkeypatch) -> None:
    monkeypatch.setattr(torch_device, "device_available", lambda name: name == "cpu")
    with pytest.raises(RuntimeError, match="'cuda' was requested"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="'cuda:1'"):
        resolve_device("cuda:1")


def test_an_available_request_is_returned_as_given(monkeypatch) -> None:
    monkeypatch.setattr(torch_device, "device_available", lambda name: True)
    assert resolve_device("cuda:1") == "cuda:1"

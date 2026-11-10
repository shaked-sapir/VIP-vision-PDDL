"""Torch device selection shared by every imaged runner."""

from __future__ import annotations

from typing import Optional


def device_available(name: str) -> bool:
    """Whether torch can place tensors on ``name`` (``cpu``, ``cuda[:N]``, ``mps``)."""
    import torch

    kind = name.split(":", 1)[0]
    if kind == "cpu":
        return True
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    return False


def resolve_device(requested: Optional[str], prefer_mps: bool = False) -> str:
    """The device a run trains on.

    ``None`` autodetects: ``cuda`` when available, then ``mps`` only when
    ``prefer_mps`` is set, else ``cpu``. A requested device is returned as given
    once checked, so a job asked for a GPU fails at start-up rather than
    training on the CPU for the whole budget.

    Raises:
        RuntimeError: if ``requested`` names a device torch cannot use here.
    """
    if requested is None:
        if device_available("cuda"):
            return "cuda"
        if prefer_mps and device_available("mps"):
            return "mps"
        return "cpu"
    if not device_available(requested):
        raise RuntimeError(
            f"device {requested!r} was requested but is not available on this host"
        )
    return requested

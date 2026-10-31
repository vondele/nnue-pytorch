"""Row flags for a gradient whose unflagged rows are exactly +0.

The tiled FT backward allocates the FT weight gradient with ``torch.zeros`` and adds only to the rows of
the batch's active features; it records those rows in a byte mask and registers (gradient, mask) here.
The merged-weight backward registers each parameter's row slice of it, and the fused RangerLite update
looks a parameter's ``.grad`` up: for an unflagged row it uses +0 instead of reading the gradient (the
same value), and skips elements whose update is the identity.

A lookup succeeds for a tensor that aliases a registered one: same data pointer, shape and version
counter. (Under torch.compile the gradient reaches the parameter as a regenerated alias, a new
TensorImpl over the same memory, so neither object identity nor a weak reference works, and a strong
reference to the tensor would make autograd copy it.) Each entry holds the gradient's storage, so the
memory cannot be freed and reused by another tensor while registered; any in-place change through the
looked-up tensor bumps its version and makes the optimizer read the gradient again. Registrations are
dropped by the next backward's ``reset`` and by ``clear()`` at the end of the optimizer step.
"""
from __future__ import annotations

import torch

_entries: list = []  # (storage, data_ptr, shape, version, flags)


def register(grad: torch.Tensor, flags: torch.Tensor, reset: bool = False) -> None:
    """Record that the rows of 2D ``grad`` with ``flags[r] == 0`` are exactly +0. ``reset`` drops all
    earlier registrations (one per backward)."""
    global _entries
    if reset:
        _entries = []
    if grad.dim() == 2 and grad.dtype == torch.float32 and grad.is_contiguous() and flags.numel() == grad.shape[0]:
        _entries.append((grad.untyped_storage(), grad.data_ptr(), tuple(grad.shape), grad._version, flags))


def clear() -> None:
    global _entries
    _entries = []


def lookup(g: torch.Tensor):
    """The row flags of ``g`` if it aliases a registered gradient (unchanged since), else None."""
    if g.dtype != torch.float32 or not g.is_contiguous():
        return None
    for _storage, ptr, shape, version, flags in _entries:
        if g.data_ptr() == ptr and tuple(g.shape) == shape:
            return flags if g._version == version else None
    return None


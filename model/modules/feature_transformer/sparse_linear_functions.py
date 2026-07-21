import torch
import torch.nn.functional as F


def _torch_sparse_linear(feature_indices, weight, bias):
    """Device-agnostic fallback for SparseLinearFunction.

    Computes: output[b] = sum_k(weight[indices[b,k]]) + bias
    Negative entries in feature_indices are treated as padding and
    contribute nothing to the sum. Uses F.embedding_bag for memory efficiency.
    """
    batch_size, max_active = feature_indices.shape
    mask = feature_indices >= 0

    if feature_indices.device.type == "mps":
        safe_indices = feature_indices.clamp(min=0).long().reshape(-1)
        per_sample_weights = mask.to(weight.dtype).reshape(-1, 1)
        gathered_weight = F.embedding(safe_indices, weight)
        output = (gathered_weight * per_sample_weights).reshape(
            batch_size, max_active, weight.shape[1]
        ).sum(dim=1)
        return output + bias

    safe_indices = feature_indices.clamp(min=0).long().reshape(-1)
    per_sample_weights = mask.to(weight.dtype).reshape(-1)
    offsets = torch.arange(
        0,
        batch_size * max_active,
        max_active,
        device=feature_indices.device,
    )
    output = F.embedding_bag(
        safe_indices,
        weight,
        offsets,
        mode="sum",
        per_sample_weights=per_sample_weights,
    )
    return output + bias


class SparseLinearFunction:
    """
    PyTorch-only sparse linear reduction. The legacy CuPy CUDA kernel was
    removed when the fused feature-transform path moved to Triton.
    """
    @staticmethod
    def apply(feature_indices, weight, bias, backend: str = "auto"):
        if backend not in ("auto", "sparse", "torch"):
            raise ValueError(f"Invalid SparseLinear backend requested: {backend}")
        return _torch_sparse_linear(feature_indices, weight, bias)

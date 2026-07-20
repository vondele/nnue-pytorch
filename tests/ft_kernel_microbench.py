"""
Isolated microbenchmark for the fused double feature-transform kernel.

Run from the repo root:
    python tests/ft_kernel_microbench.py

Useful flags:
    --batch-size 65536 --l1 1024 --repeats 50 --padding-ratio 0.2
"""

import argparse
import os
import sys
import time

# Make imports work when running from tmp/
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

import torch
import torch.cuda.nvtx as nvtx

from model.model import NNUEModel
from model.config import ModelConfig
from model.modules.feature_transformer.double_ft_functions import double_feature_transform


def make_parser():
    parser = argparse.ArgumentParser(description="Benchmark fused feature-transform kernel")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--l1", type=int, default=1024)
    parser.add_argument("--l2", type=int, default=32)
    parser.add_argument("--l3", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--padding-ratio", type=float, default=0.2,
                        help="Fraction of index slots to mark as -1 padding")
    parser.add_argument("--fake-quantize", action="store_true", default=True,
                        help="Apply fake quantization to weights/bias (training mode)")
    parser.add_argument("--no-fake-quantize", dest="fake_quantize", action="store_false")
    parser.add_argument("--feature-name", type=str, default="Full_Threats+HalfKAv2_hm^")
    return parser


def build_inputs(batch_size, feature_name, l1, fake_quantize, padding_ratio, device="cuda"):
    config = ModelConfig(L1=l1, L2=32, L3=32)
    model = NNUEModel(feature_name, config, num_psqt_buckets=8, num_ls_buckets=8).to(device)
    model.train()

    merged, bias = model.input.merged_weight_and_bias(fake_quantize_weights=fake_quantize)
    # Make them leaf Parameters so the autograd graph is exactly the same as in
    # training; otherwise the backward also differentiates the merged_weight_and_bias
    # construction and skews the benchmark.
    merged = torch.nn.Parameter(merged.to(device).detach().clone())
    bias = torch.nn.Parameter(bias.to(device).detach().clone())

    num_inputs = model.input.NUM_INPUTS
    max_active = model.input.MAX_ACTIVE_FEATURES
    output_size = bias.shape[0]
    ft_max_act = model.quantization.max_ft_activation

    us = torch.rand(batch_size, device=device)
    them = 1.0 - us
    pc = torch.randint(1, 32, (batch_size,), dtype=torch.int64, device=device)

    # Random active indices, then mark the last fraction as padding
    wi = torch.randint(0, num_inputs, (batch_size, max_active), dtype=torch.int32, device=device)
    bi = torch.randint(0, num_inputs, (batch_size, max_active), dtype=torch.int32, device=device)
    pad_count = int(max_active * padding_ratio)
    if pad_count > 0:
        wi[:, -pad_count:] = -1
        bi[:, -pad_count:] = -1

    avg_active = max_active - pad_count
    return {
        "us": us,
        "them": them,
        "wi": wi,
        "bi": bi,
        "pc": pc,
        "merged": merged,
        "bias": bias,
        "ft_max_act": ft_max_act,
        "l1_size": l1,
        "output_size": output_size,
        "num_inputs": num_inputs,
        "max_active": max_active,
        "avg_active": avg_active,
        "model": model,
    }


def bytes_forward(batch_size, avg_active, output_size, l1_size, max_active):
    """Rough byte count for the forward kernel."""
    # Index reads
    idx_bytes = 2 * batch_size * max_active * 4
    # Per active index: one weight row is read for each perspective
    weight_bytes = 2 * batch_size * avg_active * output_size * 4
    # Bias read (cached, but count it once per block)
    bias_bytes = output_size * 4
    # Outputs
    l0_bytes = batch_size * l1_size * 4
    psqt_bytes = 2 * batch_size * 4
    return idx_bytes + weight_bytes + bias_bytes + l0_bytes + psqt_bytes


def bytes_backward(batch_size, avg_active, output_size, l1_size, max_active):
    """Rough byte count for the backward kernel."""
    fwd = bytes_forward(batch_size, avg_active, output_size, l1_size, max_active)
    # Gradient outputs fed into the backward kernel
    grad_out_bytes = batch_size * l1_size * 4 + 2 * batch_size * 4
    # Atomic grad_weight updates: read-modify-write a weight row per active index
    # (two perspectives). This is an upper bound.
    grad_weight_bytes = 2 * (2 * batch_size * avg_active * output_size * 4)
    grad_bias_bytes = 2 * output_size * 4
    return fwd + grad_out_bytes + grad_weight_bytes + grad_bias_bytes


def benchmark(name, fn, repeats, warmup, bytes_per_call):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()

    ms_total = start.elapsed_time(end)
    ms_per_call = ms_total / repeats
    gb_per_s = (bytes_per_call / ms_per_call) / 1e6
    print(f"{name:30s}  {ms_per_call:7.3f} ms/call  {gb_per_s:6.1f} GB/s  ({ms_total:7.1f} ms total)")
    return ms_per_call


def main():
    args = make_parser().parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda")

    inputs = build_inputs(
        args.batch_size, args.feature_name, args.l1,
        args.fake_quantize, args.padding_ratio, device
    )

    print(f"Feature: {args.feature_name}")
    print(f"NUM_INPUTS: {inputs['num_inputs']}")
    print(f"MAX_ACTIVE_FEATURES: {inputs['max_active']}")
    print(f"avg_active (no padding): {inputs['avg_active']}")
    print(f"batch_size: {args.batch_size}, L1: {args.l1}, output_size: {inputs['output_size']}")
    print(f"fake_quantize: {args.fake_quantize}, padding_ratio: {args.padding_ratio}")
    print()

    def forward_fn():
        nvtx.range_push("ft_forward")
        l0_, wpsqt, bpsqt = double_feature_transform(
            inputs["us"], inputs["them"],
            inputs["wi"], inputs["bi"], inputs["pc"],
            inputs["merged"], inputs["bias"],
            inputs["ft_max_act"], inputs["l1_size"],
            backend="fused",
        )
        nvtx.range_pop()
        return l0_

    def fwd_bwd_fn():
        nvtx.range_push("ft_forward")
        l0_, wpsqt, bpsqt = double_feature_transform(
            inputs["us"], inputs["them"],
            inputs["wi"], inputs["bi"], inputs["pc"],
            inputs["merged"], inputs["bias"],
            inputs["ft_max_act"], inputs["l1_size"],
            backend="fused",
        )
        nvtx.range_pop()
        nvtx.range_push("ft_backward")
        (l0_.sum() + wpsqt.sum() + bpsqt.sum()).backward()
        nvtx.range_pop()

    def reset_grads():
        if inputs["merged"].grad is not None:
            inputs["merged"].grad.zero_()
        if inputs["bias"].grad is not None:
            inputs["bias"].grad.zero_()

    fwd_bytes = bytes_forward(
        args.batch_size, inputs["avg_active"], inputs["output_size"], args.l1, inputs["max_active"]
    )
    bwd_bytes = bytes_backward(
        args.batch_size, inputs["avg_active"], inputs["output_size"], args.l1, inputs["max_active"]
    )

    print("Benchmarking forward only...")
    with torch.no_grad():
        benchmark("fused_double_ft_forward", forward_fn, args.repeats, args.warmup, fwd_bytes)

    print("Benchmarking forward + backward...")
    def wrapped_fwd_bwd():
        reset_grads()
        fwd_bwd_fn()

    benchmark("fused_double_ft_fwd_bwd", wrapped_fwd_bwd, args.repeats, args.warmup, fwd_bytes + bwd_bytes)


if __name__ == "__main__":
    main()

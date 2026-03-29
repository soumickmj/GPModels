#!/usr/bin/env python3
"""
Efficiency Benchmark for GP-Models (Reviewer Response)
======================================================
Measures FLOPs, inference time (GPU + CPU), and peak memory for all models.
No training required - uses random weights and synthetic input.

Models benchmarked (matching classifier.py / classifier_rev_mproto.py configs):
  - GP-UNet            (model.Unet.GP_UNet)
  - GP-ShuffleUNet     (model.ShuffleUnet.GP_ShuffleUNet)
  - GP-ReconResNet     (model.ReconResNet.GP_ReconResNet)
  - ResNet18           (model.tvmodels.TVModelWrapper)
  - MProtoNet2D        (model.MProtoNet.MProtoNet2D)
  - InceptionV3        (model.InceptionNet)  [optional]

Usage:
    python efficiency_benchmark.py --device auto --input_size 240 --n_runs 200
"""

import argparse
import csv
import os
import sys
import time
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn
import torchvision.models as tv_models

warnings.filterwarnings("ignore")

# Ensure the project root is on sys.path so that `model.*` imports work
# when script is invoked from the project root directory.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class EvalWrapper(nn.Module):
    """Thin wrapper that always returns a single tensor from models
    whose eval-mode forward() returns a tuple (logits, mask).
    This prevents FLOPs profilers (thop / ptflops) from choking on
    tuple outputs while keeping full eval-mode computation graphs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]          # logits
        return out


def human_readable(n, suffix=""):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}{suffix}"
        n /= 1000
    return f"{n:.2f}P{suffix}"


@contextmanager
def track_gpu_memory(device):
    if device.type != "cuda":
        yield {"peak_mb": 0.0}
        return
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    info = {}
    yield info
    torch.cuda.synchronize(device)
    info["peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, dummy_input):
    """Estimate MACs/FLOPs. Uses an EvalWrapper to handle tuple outputs."""
    wrapped = EvalWrapper(model)
    wrapped.eval()
    try:
        from thop import profile as thop_profile
        macs, _ = thop_profile(wrapped, inputs=(dummy_input,), verbose=False)
        return macs, macs * 2
    except ImportError:
        pass
    try:
        from ptflops import get_model_complexity_info
        macs_str, _ = get_model_complexity_info(
            wrapped, tuple(dummy_input.shape[1:]),
            as_strings=False, print_per_layer_stat=False, verbose=False)
        return macs_str, macs_str * 2
    except ImportError:
        pass
    params = count_parameters(model)
    spatial = dummy_input.shape[-1] * dummy_input.shape[-2]
    rough_macs = params * spatial
    print("  [!] No FLOPs library found; using rough parameter-based estimate.")
    return rough_macs, rough_macs * 2


def benchmark_inference(model, dummy_input, device, n_warmup=50, n_runs=200):
    """Time inference. Handles models that return tuples in eval mode."""
    import numpy as np
    model.eval()
    model.to(device)
    x = dummy_input.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            out = model(x)              # discard; just warm up
            del out
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(x)
            del out
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "throughput": float(1000.0 / np.mean(times)),
    }


def build_models(in_channels=4, n_classes=3, input_size=240):
    """
    Instantiate every model with the SAME constructor arguments used during
    training in classifier.py / classifier_rev_mproto.py (Brats20, allCont).
    Models are returned in eval mode so forward() behaves like inference.
    """
    models = []

    # ---- GP-UNet (classifier.py, network="GP_UNet", Relu="Relu") ----
    try:
        from model.Unet.GP_UNet import GP_UNet
        m = GP_UNet(
            in_channels=in_channels,
            n_classes=n_classes,
            depth=3,
            wf=6,
            up_mode="sinc",
            dropout=True,           # Brats20 default
            Relu="Relu",
            out_act="None",         # classifier.py out_act="None"
        )
        models.append(("GP-UNet", m))
        print("  [ok] GP-UNet")
    except Exception as e:
        print(f"  [skip] GP-UNet: {e}")

    # ---- GP-ShuffleUNet (classifier.py, network="GP_ShuffleUNet") ----
    try:
        from model.ShuffleUnet.GP_ShuffleUNet import GP_ShuffleUNet
        m = GP_ShuffleUNet(
            d=2,
            in_ch=in_channels,
            num_features=64,
            n_levels=3,
            out_ch=n_classes,
            dropout=True,           # Brats20 default
            out_act="None",         # classifier.py out_act="None"
        )
        models.append(("GP-ShuffleUNet", m))
        print("  [ok] GP-ShuffleUNet")
    except Exception as e:
        print(f"  [skip] GP-ShuffleUNet: {e}")

    # ---- GP-ReconResNet (classifier.py, network="GP_ReconResNet") ----
    try:
        from model.ReconResNet.GP_ReconResNet import GP_ReconResNet
        m = GP_ReconResNet(
            in_channels=in_channels,
            n_classes=n_classes,
            res_drop_prob=0.2,      # default; classifier passes dropout (bool) but model uses 0.2
            out_act="None",         # classifier.py out_act="None"
            upinterp_algo="sinc",   # upalgo="sinc"
        )
        models.append(("GP-ReconResNet", m))
        print("  [ok] GP-ReconResNet")
    except Exception as e:
        print(f"  [skip] GP-ReconResNet: {e}")

    # ---- ResNet18 via TVModelWrapper (classifier.py, network="resnet18") ----
    try:
        from model.tvmodels import TVModelWrapper
        m = TVModelWrapper(
            model=tv_models.resnet18,
            in_channels=in_channels,     # 4 for allCont
            num_classes=n_classes,
        )
        models.append(("ResNet18", m))
        print("  [ok] ResNet18 (TVModelWrapper)")
    except Exception as e:
        print(f"  [skip] ResNet18: {e}")

    # ---- MProtoNet2D (classifier_rev_mproto.py, network="MProtoNet2D") ----
    try:
        from model.MProtoNet.MProtoNet2D import MProtoNet2D
        m = MProtoNet2D(
            in_size=(in_channels, input_size, input_size),
            out_size=n_classes,
            features="resnet50_ri",
            n_layers=7,
            prototype_shape=(30, 128, 1, 1),
            f_dist="cos",
            prototype_activation_function="log",
            p_mode=5,
            topk_p=1,
            init_weights=True,
        )
        models.append(("MProtoNet2D", m))
        print("  [ok] MProtoNet2D")
    except Exception as e:
        print(f"  [skip] MProtoNet2D: {e}")

    # ---- InceptionV3 via InceptionNet wrapper (classifier.py, network="Inception_v3") ----
    try:
        from model.InceptionNet import InceptionNet
        m = InceptionNet(
            model=tv_models.inception_v3,
            in_channels=in_channels,
            num_classes=n_classes,
        )
        models.append(("InceptionV3", m))
        print("  [ok] InceptionV3 (InceptionNet wrapper)")
    except Exception as e:
        print(f"  [skip] InceptionV3: {e}")

    return models


def write_latex_table(rows, path):
    with open(path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Computational efficiency of classification models "
                "(single 240$\\times$240 slice, batch size 1).}\n")
        f.write("\\label{tab:efficiency}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Model & Params & FLOPs & "
                "GPU (ms) & GPU Mem (MB) & CPU (ms) & Throughput \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['model']} & {r['params_h']} & {r['flops_h']} & "
                f"{r['gpu_mean']:.1f}$\\pm${r['gpu_std']:.1f} & "
                f"{r['gpu_mem']:.0f} & "
                f"{r['cpu_mean']:.1f}$\\pm${r['cpu_std']:.1f} & "
                f"{r['throughput']:.0f} sl/s \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    parser = argparse.ArgumentParser(description="Efficiency benchmark for GP models")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input_size", type=int, default=240)
    parser.add_argument("--in_channels", type=int, default=4)
    parser.add_argument("--n_classes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=200)
    parser.add_argument("--output_dir", default="reviewer_analysis/results")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'='*70}")
    print(f"  Efficiency Benchmark")
    print(f"  Device     : {device}")
    print(f"  Input      : ({args.batch_size}, {args.in_channels}, "
          f"{args.input_size}, {args.input_size})")
    print(f"  Timed runs : {args.n_runs}")
    print(f"{'='*70}\n")

    print("Loading models...")
    models = build_models(args.in_channels, args.n_classes, args.input_size)
    if not models:
        print("ERROR: No models could be loaded. Run from the project root.")
        sys.exit(1)

    dummy = torch.randn(args.batch_size, args.in_channels,
                        args.input_size, args.input_size)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for name, model in models:
        print(f"\n--- {name} ---")
        model.eval()   # eval mode: GP models produce (logits, mask) tuples
        params = count_parameters(model)
        print(f"  Parameters : {human_readable(params)}")
        try:
            macs, flops = estimate_flops(model, dummy)   # uses EvalWrapper internally
            print(f"  MACs       : {human_readable(macs)}")
            print(f"  FLOPs      : {human_readable(flops)}")
        except Exception as e:
            print(f"  FLOPs      : failed ({e})")
            macs, flops = 0, 0

        gpu_stats = {"mean_ms": 0, "std_ms": 0, "median_ms": 0,
                     "p95_ms": 0, "throughput": 0}
        gpu_mem = 0.0
        if device.type == "cuda":
            with track_gpu_memory(device) as mem_info:
                gpu_stats = benchmark_inference(model, dummy, device,
                                                n_runs=args.n_runs)
            gpu_mem = mem_info["peak_mb"]
            print(f"  GPU mean   : {gpu_stats['mean_ms']:.2f} +/- "
                  f"{gpu_stats['std_ms']:.2f} ms")
            print(f"  GPU median : {gpu_stats['median_ms']:.2f} ms")
            print(f"  GPU p95    : {gpu_stats['p95_ms']:.2f} ms")
            print(f"  GPU memory : {gpu_mem:.0f} MB")
            print(f"  Throughput : {gpu_stats['throughput']:.0f} slices/sec")

        cpu_device = torch.device("cpu")
        cpu_stats = benchmark_inference(model, dummy, cpu_device,
                                        n_warmup=10, n_runs=min(args.n_runs, 50))
        print(f"  CPU mean   : {cpu_stats['mean_ms']:.2f} +/- "
              f"{cpu_stats['std_ms']:.2f} ms")

        rows.append({
            "model": name, "params": params, "params_h": human_readable(params),
            "macs": macs, "flops": flops, "flops_h": human_readable(flops),
            "gpu_mean": gpu_stats["mean_ms"], "gpu_std": gpu_stats["std_ms"],
            "gpu_median": gpu_stats["median_ms"], "gpu_p95": gpu_stats["p95_ms"],
            "gpu_mem": gpu_mem, "cpu_mean": cpu_stats["mean_ms"],
            "cpu_std": cpu_stats["std_ms"],
            "throughput": gpu_stats["throughput"] if device.type == "cuda"
                         else cpu_stats["throughput"],
        })

    csv_path = os.path.join(args.output_dir, "efficiency_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {csv_path}")

    tex_path = os.path.join(args.output_dir, "efficiency_table.tex")
    write_latex_table(rows, tex_path)
    print(f"[saved] {tex_path}")

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    for r in rows:
        print(f"  {r['model']:20s}  params={r['params_h']:>8s}  "
              f"FLOPs={r['flops_h']:>8s}  "
              f"GPU={r['gpu_mean']:6.1f}ms  CPU={r['cpu_mean']:6.1f}ms")
    print()


if __name__ == "__main__":
    main()

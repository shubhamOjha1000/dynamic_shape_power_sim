#!/usr/bin/env python3
"""
run_demo.py -- dynamic (batch, seq_len, mode) -> time and average power.

    python run_demo.py                          # 40 random shapes, analytic model
    python run_demo.py --n 100 --seed 3
    python run_demo.py --sweep                  # deterministic grid instead
    python run_demo.py --pkg-path ../EnergAIzer/energaizer-ispass26-artifact-main \
                       --lut-dir  ../EnergAIzer/energaizer-ispass26-artifact-main/database/data

With `--pkg-path`/`--lut-dir` the real EnergAIzer predictor is used.  Without
them (or if the LUT is missing) it falls back to the SYNTHETIC analytic model,
loudly, so the pipeline still produces graphs.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynshape import (RandomShapeGenerator, ShapeRewriter, WorkloadConfig,
                      build_predictor, simulate, sweep)
from dynshape.plot import (plot_dashboard, plot_power_timeline,
                           plot_request_scatter, plot_shape_sweep)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", default=os.path.join(HERE, "templates", "gpt2"))
    ap.add_argument("--n", type=int, default=40, help="number of random shapes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-min", type=int, default=128)
    ap.add_argument("--seq-max", type=int, default=4096)
    ap.add_argument("--seq-round-to", type=int, default=128,
                    help="quantise sequence lengths (CUDA-graph style bucketing)")
    ap.add_argument("--decode-fraction", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true",
                    help="deterministic (batch x seqlen x mode) grid instead of random")
    ap.add_argument("--freq", type=int, default=900, help="SM clock, MHz")
    ap.add_argument("--pkg-path", default=None, help="EnergAIzer artifact root")
    ap.add_argument("--lut-dir", default=None, help="folder holding the LUT csv files")
    ap.add_argument("--force-analytic", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("1. learning the scaling law from three anchor traces ...")
    rw = ShapeRewriter.from_dir(args.templates)
    print(f"   {rw.n_kernels()} kernels per forward pass, "
          f"anchored at b{rw.b0} s{rw.s0}")
    print(f"   {len(rw.field_report())} distinct (op, field, exponent) classes learned")

    print("2. building the predictor ...")
    pred = build_predictor(pkg_path=args.pkg_path, lut_dir=args.lut_dir,
                           freq=args.freq, force_analytic=args.force_analytic)
    print(f"   backend: {pred.backend.name}")

    if args.sweep:
        print("3. deterministic shape grid ...")
        reqs = sweep([1, 4, 16], [128, 512, 1024, 2048, 4096])
    else:
        print(f"3. {args.n} random shapes (seed {args.seed}) ...")
        cfg = WorkloadConfig(seq_min=args.seq_min, seq_max=args.seq_max,
                             seq_round_to=args.seq_round_to,
                             decode_fraction=args.decode_fraction)
        reqs = RandomShapeGenerator(cfg, seed=args.seed).sample(args.n)

    for r in reqs[:8]:
        print(f"     {r.label:>26}   {r.tokens:>8,} tokens")
    if len(reqs) > 8:
        print(f"     ... {len(reqs) - 8} more")

    print("4. simulating ...")
    trace = simulate(reqs, rw, pred, progress=True)

    s = trace.summary()
    print("\n--- trace ---")
    print(f"  passes            {s['requests']}")
    print(f"  kernels           {s['kernels']:,}")
    print(f"  distinct shapes   {s['distinct_shapes']:,}   (cache hit rate "
          f"{s['hit_rate']*100:.1f}%)")
    print(f"  total time        {s['total_time_ms']:.2f} ms")
    print(f"  total energy      {s['total_energy_j']:.3f} J")
    print(f"  average power     {s['avg_power_w']:.1f} W")
    print(f"  peak power        {s['peak_power_w']:.1f} W")
    if trace.skipped_ops:
        print(f"  SKIPPED ops       {trace.skipped_ops}")
    if not s["is_measured_model"]:
        print("  NOTE: numbers are from the SYNTHETIC analytic model, not EnergAIzer.")

    print("\n  per pass (time, average power):")
    print(f"  {'shape':>26} {'time ms':>10} {'avg W':>8} {'J/token mJ':>12}")
    for r in trace.requests[:12]:
        print(f"  {r.mode[0]}: b{r.batch} s{r.seqlen}".rjust(28)
              + f"{r.time_ms:>10.3f} {r.avg_power_w:>8.1f} {r.energy_per_token_mj:>12.4f}")

    print("\n5. figures ...")
    for name, fn in (("dashboard", lambda: plot_dashboard(trace)),
                     ("timeline", lambda: (plot_power_timeline(trace), plt.gcf())[1]),
                     ("scatter", lambda: (plot_request_scatter(trace), plt.gcf())[1])):
        fig = fn()
        path = os.path.join(args.outdir, f"{name}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"   wrote {path}")

    if args.sweep:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        plot_shape_sweep(trace, axes[0], axes[1])
        path = os.path.join(args.outdir, "sweep.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"   wrote {path}")


if __name__ == "__main__":
    main()

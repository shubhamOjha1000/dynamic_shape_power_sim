"""
dynshape -- dynamic (batch, seq_len, mode) shapes for EnergAIzer.

EnergAIzer evaluates a *traced* workload file, so its input shape is fixed at
whatever somebody ran on a real GPU.  This package derives the kernel list for
**any** (batch, seq_len, mode) from three anchor traces, predicts each kernel,
and joins time and average power into a trace you can plot.

    from dynshape import ShapeRewriter, RandomShapeGenerator, build_predictor, simulate
    from dynshape.plot import plot_dashboard

    rw    = ShapeRewriter.from_dir("templates/gpt2")
    reqs  = RandomShapeGenerator(seed=0).sample(40)
    pred  = build_predictor(force_analytic=True)
    trace = simulate(reqs, rw, pred)
    plot_dashboard(trace)
"""

from .template import (
    ShapeRewriter,
    learn_scaling,
    load_template,
    parse_shape_from_name,
    rewrite_dims,
    rewrite_dims_qk,
    split_seq_exponent,
)
from .workload import Request, RandomShapeGenerator, WorkloadConfig, sweep
from .predictor import (
    AnalyticBackend,
    CachedPredictor,
    GeeBackend,
    build_predictor,
)
from .simulate import KernelRecord, RequestRecord, Trace, simulate

__version__ = "0.1.0"

__all__ = [
    "ShapeRewriter", "learn_scaling", "load_template", "parse_shape_from_name",
    "rewrite_dims", "rewrite_dims_qk", "split_seq_exponent",
    "Request", "RandomShapeGenerator", "WorkloadConfig", "sweep",
    "AnalyticBackend", "CachedPredictor", "GeeBackend", "build_predictor",
    "KernelRecord", "RequestRecord", "Trace", "simulate",
]

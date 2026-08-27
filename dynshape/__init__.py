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
from .workload import (Request, RandomShapeGenerator, WorkloadConfig,
                       conversation, generation, sweep)
from .predictor import (
    AnalyticBackend,
    CachedPredictor,
    GeeBackend,
    build_predictor,
)
from .simulate import KernelRecord, RequestRecord, Trace, simulate

# -- L0/L1: a real serving engine ------------------------------------------
#
# Everything above this line prices a *shape*.  Everything below decides which
# shapes a real engine would actually produce, and when.
#
#     from dynshape import (TrafficConfig, generate_traffic, SchedulerConfig,
#                           EngineConfig, run_engine)
#     from dynshape.engine_plot import plot_engine_dashboard
#
#     reqs  = generate_traffic(TrafficConfig(interval="gamma", qps=6, cv=2.0,
#                                            length="zipf", num_requests=200))
#     rw    = ShapeRewriter.from_dir("templates/gpt2")
#     pred  = build_predictor(force_analytic=True)
#     trace = run_engine(reqs, rw, pred, EngineConfig())
#     plot_engine_dashboard(trace)

from .arrival import (GammaInterval, IntervalGenerator, PoissonInterval,
                      StaticInterval, TraceInterval, piecewise_poisson)
from .lengths import (FixedLength, LengthGenerator, TraceLength, UniformLength,
                      ZipfGenerator, ZipfLength)
from .entities import Batch, Piece, SimRequest, reset_ids
from .traffic import TrafficConfig, generate_traffic, spawn_seeds, traffic_summary
from .kvcache import (BlockAllocator, HardwareConfig, MemoryPlanner, ModelConfig)
from .scheduler import ChunkedPrefillScheduler, SchedulerConfig
from .mixed import (attention_mask, build_iteration_kernels,
                    build_iteration_kernels_tagged, fusible_mask,
                    iteration_token_shapes, mixed_report)
from .work import WORK_FIELDS, gemm_work, kernel_work
from .measure import (PowerSampler, bin_mean, compare_to_trace, idle_baseline)
from .replay import (GPT2_MAX_POSITIONS, ReplayRequest, build_replay_traffic,
                     check_fits_context, load_spec, prompt_token_ids, save_spec,
                     spec_summary, to_spec)
from .energaizer import (ArtifactPaths, build_estimator, build_gee_predictor,
                         flatten_lut, idle_power_table, measured_idle_power_w,
                         clone_artifact, download_lut, locate_artifact,
                         lut_status)
from .engine import EngineConfig, EngineTrace, IterationRecord, Segment, run_engine

__version__ = "0.2.0"

__all__ = [
    # L2 -- dynamic shapes
    "ShapeRewriter", "learn_scaling", "load_template", "parse_shape_from_name",
    "rewrite_dims", "rewrite_dims_qk", "split_seq_exponent",
    "Request", "RandomShapeGenerator", "WorkloadConfig",
    "conversation", "generation", "sweep",
    "AnalyticBackend", "CachedPredictor", "GeeBackend", "build_predictor",
    "KernelRecord", "RequestRecord", "Trace", "simulate",
    # L0 -- workload
    "IntervalGenerator", "StaticInterval", "PoissonInterval", "GammaInterval",
    "TraceInterval", "piecewise_poisson",
    "LengthGenerator", "FixedLength", "UniformLength", "ZipfLength",
    "TraceLength", "ZipfGenerator",
    "TrafficConfig", "generate_traffic", "traffic_summary", "spawn_seeds",
    # L1 -- serving engine
    "SimRequest", "Batch", "Piece", "reset_ids",
    "ModelConfig", "HardwareConfig", "MemoryPlanner", "BlockAllocator",
    "SchedulerConfig", "ChunkedPrefillScheduler",
    # L2 for mixed batches
    "build_iteration_kernels", "build_iteration_kernels_tagged",
    "fusible_mask", "attention_mask", "mixed_report", "iteration_token_shapes",
    # work accounting, independent of any power model
    "kernel_work", "gemm_work", "WORK_FIELDS",
    # the measured model
    "build_gee_predictor", "clone_artifact", "download_lut", "locate_artifact",
    "lut_status", "build_estimator", "ArtifactPaths",
    "flatten_lut", "measured_idle_power_w", "idle_power_table",
    # measuring a real GPU, and replaying traffic through a real engine
    "PowerSampler", "idle_baseline", "bin_mean", "compare_to_trace",
    "build_replay_traffic", "to_spec", "save_spec", "load_spec", "spec_summary",
    "prompt_token_ids", "check_fits_context", "ReplayRequest", "GPT2_MAX_POSITIONS",
    # the loop
    "EngineConfig", "EngineTrace", "IterationRecord", "Segment", "run_engine",
]

"""
engine_plot.py
==============

Figures for an `EngineTrace`.  Separate from `plot.py`, which draws the
shape-stream trace and knows nothing about arrivals, queues or KV.

EVERY FIGURE STAMPS ITS BACKEND.  A roofline stand-in and a measured LUT
produce plots that look identical, and a figure that does not say which one it
came from is a figure that will eventually be quoted as a measurement.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .engine import EngineTrace


def _backend_note(trace: EngineTrace) -> str:
    name = trace.predictor_stats.get("backend", "unknown")
    measured = trace.predictor_stats.get("is_measured_model", False)
    return f"predictor: {name}" + ("" if measured else "   *** SYNTHETIC -- not a measurement ***")


def _stamp(fig, trace: EngineTrace) -> None:
    fig.text(0.005, 0.005, _backend_note(trace), fontsize=7,
             color="#444444", ha="left", va="bottom")


def plot_power_timeline(trace: EngineTrace, ax=None, max_ms: Optional[float] = None,
                        show_idle: bool = True, mark_preemptions: bool = True):
    """Average power per iteration against wall clock.

    Idle stretches are shaded rather than merely drawn low, because the eye
    reads a low flat line as "running quietly" when it actually means "not
    running at all" -- and the distinction is the whole duty-cycle story.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))

    limit = max_ms if max_ms is not None else trace.total_time_ms
    its = [i for i in trace.iterations if i.t_start_ms < limit]

    ts, ps = [], []
    for i in its:
        ts.append(i.t_start_ms)
        ps.append(i.avg_power_w)
    if its:
        last = its[-1]
        ts.append(min(last.t_end_ms, limit))
        ps.append(last.avg_power_w)

    if show_idle:
        for s in trace.idle_segments:
            if s.t_start_ms >= limit:
                continue
            ax.axvspan(s.t_start_ms, min(s.t_end_ms, limit),
                       color="#dbe4ee", alpha=0.7, lw=0, zorder=0)

    ax.step(ts, ps, where="post", lw=0.9, color="#1f4e79", zorder=3)
    ax.fill_between(ts, ps, step="post", alpha=0.18, color="#1f4e79", zorder=2)

    if mark_preemptions:
        prev = 0
        first = True
        for i in its:
            if i.preemptions_total > prev:
                ax.axvline(i.t_start_ms, color="#c0392b", lw=0.8, alpha=0.75,
                           zorder=4, label="preemption" if first else None)
                first = False
                prev = i.preemptions_total

    ax.axhline(trace.avg_power_w, color="#e67e22", ls="--", lw=1.0,
               label=f"mean over wall clock {trace.avg_power_w:.0f} W", zorder=5)
    ax.axhline(trace.avg_busy_power_w, color="#16a085", ls=":", lw=1.0,
               label=f"mean while busy {trace.avg_busy_power_w:.0f} W", zorder=5)

    ax.set_xlim(0, limit)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("wall clock (ms)")
    ax.set_ylabel("average power (W)")
    title = "Power over time"
    if max_ms is not None:
        title += f" -- first {max_ms:.0f} ms"
    ax.set_title(f"{title}   (shaded = idle, duty cycle {trace.duty_cycle:.0%})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    return ax


def plot_batch_composition(trace: EngineTrace, ax=None, max_ms: Optional[float] = None):
    """Tokens per iteration, split into prefill and decode.

    This is the picture of chunked prefill actually working: a band of decode
    tokens that never disappears, with prefill filling whatever budget is left
    above it.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.4))

    limit = max_ms if max_ms is not None else trace.total_time_ms
    its = [i for i in trace.iterations if i.t_start_ms < limit]
    if not its:
        return ax

    t = [i.t_start_ms for i in its]
    dec = [i.decode_tokens for i in its]
    pre = [i.prefill_tokens for i in its]
    tot = [d + p for d, p in zip(dec, pre)]

    ax.fill_between(t, 0, dec, step="post", color="#2980b9", alpha=0.85,
                    label="decode tokens (1 per running request)")
    ax.fill_between(t, dec, tot, step="post", color="#e67e22", alpha=0.85,
                    label="prefill chunk tokens")

    budget = trace.scheduler_report.get("chunk_size")
    if budget:
        ax.axhline(budget, color="#555555", ls="--", lw=0.9,
                   label=f"token budget {budget}")

    mixed = sum(1 for i in its if i.is_mixed)
    ax.set_xlim(0, limit)
    ax.set_xlabel("wall clock (ms)")
    ax.set_ylabel("tokens in the iteration")
    ax.set_title(f"Batch composition -- {mixed}/{len(its)} iterations "
                 f"({mixed / len(its):.0%}) carry prefill and decode together")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    return ax


def plot_queue_and_kv(trace: EngineTrace, ax=None, max_ms: Optional[float] = None):
    """KV utilisation and queue depth -- the two things that cause preemption."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.4))

    limit = max_ms if max_ms is not None else trace.total_time_ms
    its = [i for i in trace.iterations if i.t_start_ms < limit]
    if not its:
        return ax

    t = [i.t_start_ms for i in its]
    ax.plot(t, [100 * i.kv_utilisation for i in its], lw=1.1, color="#8e44ad",
            label="KV pool used (%)")
    ax.set_ylabel("KV pool used (%)", color="#8e44ad")
    ax.tick_params(axis="y", labelcolor="#8e44ad")
    ax.set_ylim(0, max(101, 1.1 * max(100 * i.kv_utilisation for i in its)))

    ax2 = ax.twinx()
    ax2.plot(t, [i.n_running for i in its], lw=1.0, color="#16a085",
             label="running")
    ax2.plot(t, [i.n_waiting for i in its], lw=1.0, color="#c0392b", ls="--",
             label="waiting")
    ax2.set_ylabel("requests", color="#333333")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left",
              fontsize=8, framealpha=0.9)
    ax.set_xlim(0, limit)
    ax.set_xlabel("wall clock (ms)")
    ax.set_title("Queue depth and KV pressure")
    ax.grid(alpha=0.25)
    return ax


def plot_request_latency(trace: EngineTrace, ax=None):
    """Time to first token against arrival, sized by prompt.

    The shape to look for: TTFT climbing with arrival time means the queue is
    growing and the run never reached steady state, so its average power is a
    transient, not a rate.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    done = [r for r in trace.requests if r.ttft_s is not None]
    if not done:
        return ax

    x = [r.arrived_at for r in done]
    y = [r.ttft_s for r in done]
    s = [max(8, min(160, r.num_prefill_tokens / 12)) for r in done]
    c = ["#c0392b" if r.num_restarts else "#1f4e79" for r in done]

    ax.scatter(x, y, s=s, c=c, alpha=0.65, edgecolors="none")
    ax.set_xlabel("arrival time (s)")
    ax.set_ylabel("time to first token (s)")
    n_restart = sum(1 for r in done if r.num_restarts)
    note = f", {n_restart} restarted (red)" if n_restart else ""
    ax.set_title(f"TTFT vs arrival -- marker size = prompt length{note}")
    ax.grid(alpha=0.25)
    return ax


def plot_energy_split(trace: EngineTrace, ax=None):
    """Where the joules went: fused linear layers vs per-request attention.

    Only the attention half is phase-attributable.  A fused GEMM genuinely
    belongs to prefill and decode at once, which is the point of fusing it, so
    it gets its own bar rather than being split by some arbitrary rule.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))

    fused = sum(i.energy_fused_j for i in trace.iterations)
    attn_p = sum(i.energy_attn_prefill_j for i in trace.iterations)
    attn_d = sum(i.energy_attn_decode_j for i in trace.iterations)
    idle = trace.idle_energy_j

    labels = ["fused\n(linear, MLP,\nnorms)", "attention\nprefill",
              "attention\ndecode", "idle\n(gap + waiting)"]
    values = [fused, attn_p, attn_d, idle]
    colors = ["#34495e", "#e67e22", "#2980b9", "#95a5a6"]

    bars = ax.bar(labels, values, color=colors)
    total = sum(values) or 1.0
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.1f} J\n{v / total:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("energy (J)")
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.set_title("Energy attribution")
    ax.grid(alpha=0.25, axis="y")
    return ax


def plot_kernel_zoom(trace: EngineTrace, ax=None, max_ms: Optional[float] = None):
    """The kernel-resolution staircase, where kernels were recorded.

    This is the granularity the whole project is built for; it is only kept for
    the first slice of the run because a busy engine launches millions of them.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.6))

    if not trace.segments:
        ax.text(0.5, 0.5, "no per-kernel records\n(record_kernels_until_ms = 0)",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    # The recording window opens at the first iteration, not at t=0 -- under
    # light load the engine idles first, so anchoring the axis at zero would
    # squeeze every recorded kernel into a sliver at the right-hand edge.
    start = min(s.t_start_ms for s in trace.segments)
    recorded = max(s.t_end_ms for s in trace.segments)
    limit = min(start + max_ms, recorded) if max_ms is not None else recorded
    segs = sorted((s for s in trace.segments if s.t_start_ms < limit),
                  key=lambda s: s.t_start_ms)

    ts = [s.t_start_ms for s in segs]
    ps = [s.power_w for s in segs]
    if segs:
        ts.append(min(segs[-1].t_end_ms, limit))
        ps.append(segs[-1].power_w)

    ax.step(ts, ps, where="post", lw=0.7, color="#1f4e79")
    ax.fill_between(ts, ps, step="post", alpha=0.2, color="#1f4e79")

    # Mark where each iteration starts, so the per-pass rhythm is visible.
    for i in trace.iterations:
        if i.t_start_ms >= limit:
            break
        if i.t_start_ms >= start:
            ax.axvline(i.t_start_ms, color="#aaaaaa", lw=0.4, alpha=0.6, zorder=0)

    ax.set_xlim(start, limit)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("wall clock (ms)")
    ax.set_ylabel("power (W)")
    ax.set_title(f"Kernel-level power -- {len(segs)} kernels over "
                 f"{limit - start:.0f} ms from the first iteration "
                 f"(grey lines = iteration boundaries)")
    ax.grid(alpha=0.25)
    return ax


def plot_engine_dashboard(trace: EngineTrace, max_ms: Optional[float] = None,
                          zoom_ms: Optional[float] = None, figsize=(15, 15)):
    """Six panels: the whole run, then a kernel-level zoom into its opening."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.15, 0.95, 0.95, 1.0, 1.1],
                          hspace=0.55, wspace=0.25)

    plot_power_timeline(trace, ax=fig.add_subplot(gs[0, :]), max_ms=max_ms)
    plot_batch_composition(trace, ax=fig.add_subplot(gs[1, :]), max_ms=max_ms)
    plot_queue_and_kv(trace, ax=fig.add_subplot(gs[2, :]), max_ms=max_ms)
    plot_request_latency(trace, ax=fig.add_subplot(gs[3, 0]))
    plot_energy_split(trace, ax=fig.add_subplot(gs[3, 1]))
    plot_kernel_zoom(trace, ax=fig.add_subplot(gs[4, :]),
                     max_ms=zoom_ms if zoom_ms is not None
                     else trace.kernels_truncated_at_ms)

    s = trace.summary()
    fig.suptitle(
        f"Serving-engine power trace -- {s['requests']} requests, "
        f"{s['iterations']} iterations ({s['mixed_fraction']:.0%} mixed), "
        f"{s['wall_time_s']:.2f} s, {s['total_energy_j']:.1f} J, "
        f"{s['avg_power_w_wallclock']:.0f} W wall / {s['avg_power_w_busy']:.0f} W busy",
        fontsize=12, y=0.995)
    _stamp(fig, trace)
    return fig


def plot_resampled_power(trace: EngineTrace, dt_ms: float = 1.0,
                         smooth_tau_ms: Optional[float] = 5.0,
                         ax=None, max_ms: Optional[float] = None,
                         show_raw: bool = True):
    """The trace on a fixed time grid -- what a meter would actually record.

    Three things overlaid, deliberately:

      * the event-based staircase (variable-width iterations) in grey
      * the resampled series at `dt_ms`, energy-conserving
      * the same series through a one-pole board response at `smooth_tau_ms`

    The gap between the first and the third is the answer to "would a sensor see
    this?", and it is usually large. Assumption 5 of the design doc says a real
    sensor never sees the square edges a kernel trace produces; this is that
    assumption made visible rather than asserted.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))

    limit = max_ms if max_ms is not None else trace.total_time_ms

    if show_raw:
        ts, ps = trace.power_steps()
        ax.step(ts, ps, where="post", lw=0.6, color="#bbbbbb", zorder=1,
                label="per-iteration events (variable width)")

    t, p = trace.resample(dt_ms=dt_ms)
    ax.plot(t, p, lw=0.9, color="#1f4e79", zorder=3,
            label=f"resampled, dt = {dt_ms:g} ms")

    if smooth_tau_ms:
        _, ps_s = trace.resample(dt_ms=dt_ms, smooth_tau_ms=smooth_tau_ms)
        ax.plot(t, ps_s, lw=1.6, color="#c0392b", zorder=4,
                label=f"through a {smooth_tau_ms:g} ms board response")

    # Energy conservation is the property that makes the resample trustworthy,
    # so state it on the figure rather than in a docstring nobody reads.
    integral = float(np.sum(p) * dt_ms / 1000.0)
    err = abs(integral - trace.total_energy_j) / max(trace.total_energy_j, 1e-12)

    ax.set_xlim(0, limit)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("wall clock (ms)")
    ax.set_ylabel("power (W)")
    ax.set_title(f"Fixed-rate power -- integral {integral:.1f} J vs "
                 f"{trace.total_energy_j:.1f} J from the events "
                 f"({err:.2e} relative error)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    return ax


def plot_work_vector(trace: EngineTrace, ax=None, max_ms: Optional[float] = None):
    """Arithmetic and traffic per iteration, independent of any power model.

    Reported beside the watts because it separates the two error sources: if a
    predicted trace disagrees with a measured one, this says whether the
    simulator got the *work* wrong or the *conversion to watts* wrong. With only
    watts, those are indistinguishable.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 3.4))

    limit = max_ms if max_ms is not None else trace.total_time_ms
    its = [i for i in trace.iterations if i.t_start_ms < limit]
    if not its:
        return ax

    t = [i.t_start_ms for i in its]
    lin = [i.linear_flops / 1e9 for i in its]
    att = [i.attn_flops / 1e9 for i in its]
    tot = [a + b for a, b in zip(lin, att)]

    ax.fill_between(t, 0, lin, step="post", color="#34495e", alpha=0.85,
                    label="linear / MLP GFLOP (fused)")
    ax.fill_between(t, lin, tot, step="post", color="#e67e22", alpha=0.85,
                    label="attention GFLOP (per request)")
    ax.set_xlim(0, limit)
    ax.set_xlabel("wall clock (ms)")
    ax.set_ylabel("GFLOP in the iteration")

    ax2 = ax.twinx()
    ax2.plot(t, [i.arithmetic_intensity for i in its], lw=0.8, color="#16a085")
    ax2.set_ylabel("FLOP / byte", color="#16a085")
    ax2.tick_params(axis="y", labelcolor="#16a085")

    ax.set_title("Work vector -- the arithmetic behind the watts "
                 "(green: arithmetic intensity, i.e. which side of the roofline)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    return ax

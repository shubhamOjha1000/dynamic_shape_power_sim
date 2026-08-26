"""
plot.py
=======

Join **time** and **average power** into graphs.

Three views, because they answer different questions:

  plot_power_timeline   power vs wall-clock -- the trace itself, a staircase.
                        Shows *when* the machine is hot.
  plot_request_scatter  one point per forward pass: how long it took vs what it
                        drew.  Shows *which shapes* are hot.  This is the join
                        the brief asked for.
  plot_shape_sweep      average power and time against sequence length, one
                        line per batch size.  Shows *how* power moves with the
                        dynamic inputs -- the thing a random stream alone
                        cannot make legible.

`plot_dashboard` puts all three on one figure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_MODE_COLOR = {"prefill": "#c0392b", "decode": "#2471a3"}
_SYNTH_NOTE = "SYNTHETIC analytic model -- not an EnergAIzer prediction"


def _provenance(trace) -> str:
    st = trace.predictor_stats or {}
    if st.get("is_measured_model"):
        return "EnergAIzer (LUT-calibrated)"
    return _SYNTH_NOTE


def _stamp(ax, trace):
    ax.text(0.995, 0.02, _provenance(trace), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, alpha=0.65, style="italic")


# ---------------------------------------------------------------------------


def plot_power_timeline(trace, ax=None, max_ms: Optional[float] = None,
                        shade_requests: bool = True, title: Optional[str] = None):
    """Power vs time as a staircase, with each forward pass shaded by mode."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))

    ts, ps = trace.power_steps()
    ts, ps = np.asarray(ts), np.asarray(ps)
    if max_ms is not None:
        keep = ts <= max_ms
        ts, ps = ts[keep], ps[keep]

    if shade_requests:
        for r in trace.requests:
            if max_ms is not None and r.t_start_ms > max_ms:
                break
            ax.axvspan(r.t_start_ms, r.t_start_ms + r.time_ms,
                       color=_MODE_COLOR.get(r.mode, "#888"), alpha=0.10, lw=0)

    ax.step(ts, ps, where="post", lw=0.9, color="#1b1b1b")
    ax.axhline(trace.avg_power_w, ls="--", lw=1.0, color="#e67e22",
               label=f"trace average {trace.avg_power_w:.0f} W")

    ax.set_xlabel("time (ms)")
    ax.set_ylabel("power (W)")
    ax.set_title(title or "Per-kernel power over time")
    ax.grid(alpha=0.25)

    handles = [plt.Line2D([], [], color="#e67e22", ls="--", lw=1.0,
                          label=f"trace average {trace.avg_power_w:.0f} W")]
    for mode, c in _MODE_COLOR.items():
        handles.append(matplotlib.patches.Patch(color=c, alpha=0.25, label=mode))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    _stamp(ax, trace)
    return ax


def plot_request_scatter(trace, ax=None, annotate_top: int = 4,
                         title: Optional[str] = None):
    """One point per forward pass: duration (x) against average power (y).

    Marker area is proportional to tokens processed, so the big prefills are
    visibly big.  This is the time-and-power join, per dynamic shape.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for mode in ("prefill", "decode"):
        rs = [r for r in trace.requests if r.mode == mode]
        if not rs:
            continue
        x = [r.time_ms for r in rs]
        y = [r.avg_power_w for r in rs]
        tok = np.asarray([max(r.tokens, 1) for r in rs], dtype=float)
        size = 18 + 170 * (np.log10(tok) - np.log10(tok.min() if tok.min() > 0 else 1)) / \
            max(np.log10(tok.max()) - np.log10(tok.min() if tok.min() > 0 else 1), 1e-9)
        ax.scatter(x, y, s=size, alpha=0.68, color=_MODE_COLOR[mode],
                   edgecolor="white", lw=0.5, label=f"{mode} (n={len(rs)})")

    if annotate_top and trace.requests:
        for r in sorted(trace.requests, key=lambda r: r.time_ms)[-annotate_top:]:
            ax.annotate(f"b{r.batch} s{r.seqlen}", (r.time_ms, r.avg_power_w),
                        textcoords="offset points", xytext=(6, 5), fontsize=7.5, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("forward-pass time (ms, log)")
    ax.set_ylabel("average power (W)")
    ax.set_title(title or "Time vs average power, per dynamic shape")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    _stamp(ax, trace)
    return ax


def plot_shape_sweep(trace, ax_power=None, ax_time=None, title: Optional[str] = None):
    """Average power and time against sequence length, one line per batch.

    Meant for a `workload.sweep(...)` grid, where every (batch, seqlen) pair is
    present exactly once per mode.  On a random stream the lines will be ragged.
    """
    if ax_power is None or ax_time is None:
        _, (ax_power, ax_time) = plt.subplots(1, 2, figsize=(13, 4.4))

    by: Dict = {}
    for r in trace.requests:
        by.setdefault((r.mode, r.batch), []).append(r)

    cmap = plt.get_cmap("viridis")
    batches = sorted({b for _, b in by})
    for (mode, batch), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r.seqlen)
        s = [r.seqlen for r in rs]
        c = cmap(batches.index(batch) / max(len(batches) - 1, 1))
        ls = "-" if mode == "prefill" else "--"
        ax_power.plot(s, [r.avg_power_w for r in rs], ls, marker="o", ms=3.5,
                      color=c, lw=1.4, label=f"b={batch} {mode}")
        ax_time.plot(s, [r.time_ms for r in rs], ls, marker="o", ms=3.5,
                     color=c, lw=1.4, label=f"b={batch} {mode}")

    for ax, lab in ((ax_power, "average power (W)"), (ax_time, "forward-pass time (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("sequence length (prefill) / context length (decode)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, which="both")
    ax_time.set_yscale("log")
    ax_power.set_title(title or "Average power vs shape")
    ax_time.set_title("Time vs shape")
    ax_power.legend(fontsize=6.5, ncol=2)
    _stamp(ax_power, trace)
    return ax_power, ax_time


def plot_dashboard(trace, max_ms: Optional[float] = None, suptitle: Optional[str] = None):
    """Timeline on top, scatter and per-mode distribution underneath."""
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.34, wspace=0.24)

    plot_power_timeline(trace, ax=fig.add_subplot(gs[0, :]), max_ms=max_ms)
    plot_request_scatter(trace, ax=fig.add_subplot(gs[1, 0]))

    ax = fig.add_subplot(gs[1, 1])
    data, labels, colors = [], [], []
    for mode in ("prefill", "decode"):
        vals = [r.avg_power_w for r in trace.requests if r.mode == mode]
        if vals:
            data.append(vals)
            labels.append(f"{mode}\n(n={len(vals)})")
            colors.append(_MODE_COLOR[mode])
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("#1b1b1b")
    ax.set_ylabel("average power (W)")
    ax.set_title("Power by mode")
    ax.grid(alpha=0.25, axis="y")
    _stamp(ax, trace)

    s = trace.summary()
    fig.suptitle(
        suptitle or
        f"Dynamic (batch, seq_len, mode) -> power  |  {s['requests']} passes, "
        f"{s['kernels']} kernels, {s['total_time_ms']:.1f} ms, "
        f"{s['total_energy_j']:.2f} J, avg {s['avg_power_w']:.0f} W",
        fontsize=12,
    )
    return fig

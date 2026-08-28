"""
fleet_plot.py
=============

Figures for a `FleetTrace`.  Separate from `engine_plot.py`, which draws one
GPU and knows nothing about routers.

The figure that matters is `plot_facility_power`: N replica traces stacked, with
the facility total drawn on top.  Everything the routing layer exists to show is
in the gap between the tallest single replica and that total.

EVERY FIGURE STAMPS ITS BACKEND, for the same reason `engine_plot.py` does: a
roofline stand-in and a measured LUT produce plots that look identical, and a
figure that does not say which one it came from is a figure that will eventually
be quoted as a measurement.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .fleet import FleetTrace

#: One colour per replica, colour-blind-safe, stable across every figure here
#: so replica 2 is the same colour in all of them.
REPLICA_COLOURS = ["#4477AA", "#EE6677", "#228833", "#CCBB44",
                   "#66CCEE", "#AA3377", "#BBBBBB", "#000000"]


def _colour(k: int) -> str:
    return REPLICA_COLOURS[k % len(REPLICA_COLOURS)]


def _backend_note(trace: FleetTrace) -> str:
    stats = trace.replicas[0].predictor_stats if trace.replicas else {}
    name = stats.get("backend", "unknown")
    measured = stats.get("is_measured_model", False)
    return (f"predictor: {name}   routing: {trace.routing}"
            + ("" if measured else "   *** SYNTHETIC -- not a measurement ***"))


def _stamp(fig, trace: FleetTrace) -> None:
    fig.text(0.005, 0.005, _backend_note(trace), fontsize=7,
             color="#444444", ha="left", va="bottom")


def plot_facility_power(trace: FleetTrace, dt_ms: float = 250.0, ax=None,
                        smooth_tau_ms: Optional[float] = None,
                        show_total: bool = True, stacked: bool = True):
    """**The figure.**  Per-replica power stacked, facility total on top.

    Stacked rather than overlaid on purpose: the height of the stack *is* the
    facility draw, so the eye reads the number a breaker sees instead of having
    to add four wiggling lines together.

    The idle floor is drawn as a horizontal rule at `N x P_idle`.  Everything
    below it is the cost of having the GPUs switched on, which no router can
    change; only the part above it is in play.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4.2))

    t, per_replica, total = trace.resample(dt_ms=dt_ms, smooth_tau_ms=smooth_tau_ms)
    if not total.size:
        ax.set_title("empty trace")
        return ax
    t_s = t / 1000.0

    if stacked:
        ax.stackplot(t_s, *per_replica,
                     labels=[f"replica {k}" for k in range(trace.num_replicas)],
                     colors=[_colour(k) for k in range(trace.num_replicas)],
                     alpha=0.85, linewidth=0)
    else:
        for k in range(trace.num_replicas):
            ax.plot(t_s, per_replica[k], color=_colour(k), lw=1.0,
                    label=f"replica {k}")

    if show_total and stacked:
        ax.plot(t_s, total, color="black", lw=1.2, label="facility total")

    floor = trace.idle_w * trace.num_replicas
    if floor > 0:
        ax.axhline(floor, color="#888888", ls=":", lw=1.0)
        ax.text(t_s[-1], floor, f"  {trace.num_replicas} x idle = {floor:.0f} W",
                fontsize=7, color="#666666", va="bottom", ha="right")

    c = trace.coincidence(dt_ms=dt_ms)
    if c:
        ax.axhline(c["facility_peak_w"], color="#CC0000", ls="--", lw=1.0)
        ax.text(0.0, c["facility_peak_w"],
                f" facility peak {c['facility_peak_w']:.0f} W "
                f"@ {dt_ms:.0f} ms", fontsize=8, color="#CC0000", va="bottom")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("power (W)")
    ax.set_title(f"Facility power -- {trace.num_replicas} replicas, "
                 f"routing = {trace.routing}, {dt_ms:.0f} ms aperture")
    ax.legend(fontsize=7, ncol=min(5, trace.num_replicas + 1), loc="upper right")
    ax.margins(x=0)
    ax.grid(alpha=0.25)
    return ax


def plot_replica_balance(trace: FleetTrace, ax=None):
    """Requests and tokens per replica -- the two disagree, and that is the point.

    `round_robin` is exact on counts by construction and can still be badly
    skewed on tokens, because it never looks at how big a request is.  Reading
    only the left bars is how a router gets called "perfectly balanced".
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    bal = trace.balance()
    n = trace.num_replicas
    x = np.arange(n)
    counts = np.asarray(bal["per_replica_requests"], dtype=float)
    tokens = np.asarray(bal["per_replica_tokens"], dtype=float)

    # Normalised to their own means so two different units share one axis.
    cn = counts / counts.mean() if counts.mean() else counts
    tn = tokens / tokens.mean() if tokens.mean() else tokens

    ax.bar(x - 0.19, cn, width=0.36, color="#88AACC", label="requests")
    ax.bar(x + 0.19, tn, width=0.36, color="#CC8866", label="tokens")
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"r{k}" for k in range(n)])
    ax.set_ylabel("share, relative to an even split")
    ax.set_title(f"Load balance -- {trace.routing}\n"
                 f"tokens max/mean = {bal['tokens']['max_over_mean']:.2f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    return ax


def plot_replica_duty(trace: FleetTrace, ax=None):
    """Duty cycle per replica: how much of the run each GPU spent computing.

    A router that concentrates work shows up here as a tall bar beside short
    ones -- and short bars are not free, because an idle GPU still draws its
    floor.  Concentration lowers facility *peak* by leaving hardware unused,
    which is a way of failing rather than a way of winning.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    rows = trace.per_replica_summary()
    x = np.arange(len(rows))
    ax.bar(x, [r["duty_cycle"] for r in rows],
           color=[_colour(k) for k in range(len(rows))])
    ax.set_xticks(x)
    ax.set_xticklabels([f"r{r['replica']}" for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("duty cycle")
    ax.set_title(f"Per-replica utilisation -- fleet mean "
                 f"{trace.duty_cycle:.0%}", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    return ax


def plot_aperture_sensitivity(trace: FleetTrace,
                              apertures_ms: Sequence[float] = (10, 50, 250, 1000, 5000),
                              ax=None):
    """Facility peak against meter aperture, with energy alongside as the control.

    A peak is not a number until you say over what window.  Energy is flat
    across the sweep to four significant figures -- if it is not, the resampler
    is broken -- while the peak falls monotonically.  Quoting a peak without its
    aperture leaves that whole range of ambiguity in the number.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    rows = trace.aperture_table(apertures_ms)
    xs = [r["aperture_ms"] for r in rows]
    ax.plot(xs, [r["facility_peak_w"] for r in rows], "o-", color="#CC0000",
            label="facility peak")
    ax.plot(xs, [r["facility_mean_w"] for r in rows], "s--", color="#4477AA",
            label="facility mean")
    ax.set_xscale("log")
    ax.set_xlabel("meter aperture (ms)")
    ax.set_ylabel("power (W)")
    ax.set_title(f"Peak depends on aperture -- {trace.routing}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    return ax


def plot_routing_comparison(traces: Dict[str, FleetTrace], dt_ms: float = 250.0,
                            figsize=(13, 4.2)):
    """The same traffic under several routers, facility totals overlaid.

    One line per policy.  Identical arrivals, identical prompts, identical
    output lengths -- the only difference is which replica each request landed
    on, which is what makes the gap between the lines attributable.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax, bx = axes

    for k, (name, tr) in enumerate(traces.items()):
        t, _, total = tr.resample(dt_ms=dt_ms)
        ax.plot(t / 1000.0, total, lw=1.1, color=_colour(k), label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("facility power (W)")
    ax.set_title(f"Facility power by routing policy ({dt_ms:.0f} ms aperture)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.margins(x=0)
    ax.grid(alpha=0.25)

    names = list(traces)
    peaks = [traces[n].coincidence(dt_ms=dt_ms).get("facility_peak_w", 0.0)
             for n in names]
    bx.barh(range(len(names)), peaks,
            color=[_colour(k) for k in range(len(names))])
    bx.set_yticks(range(len(names)))
    bx.set_yticklabels(names, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlabel("facility peak (W)")
    bx.set_title("what sizes the breaker", fontsize=10)
    bx.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    if traces:
        _stamp(fig, next(iter(traces.values())))
    return fig


def plot_fleet_dashboard(trace: FleetTrace, dt_ms: float = 250.0,
                         figsize=(15, 11)):
    """Four panels: the facility trace, then who did the work and how it meters."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], hspace=0.45,
                          wspace=0.28)

    plot_facility_power(trace, dt_ms=dt_ms, ax=fig.add_subplot(gs[0, :]))
    plot_replica_balance(trace, ax=fig.add_subplot(gs[1, 0]))
    plot_replica_duty(trace, ax=fig.add_subplot(gs[1, 1]))
    plot_aperture_sensitivity(trace, ax=fig.add_subplot(gs[1, 2]))

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    s = trace.summary(dt_ms=dt_ms)
    lines = [
        f"routing = {s['routing']}    replicas = {s['num_replicas']}    "
        f"requests = {s['requests']} ({s['completed']} completed)",
        f"wall {s['wall_time_s']:.2f} s    fleet duty cycle {s['fleet_duty_cycle']:.0%}    "
        f"energy {s['total_energy_j']:.1f} J    "
        f"({s['tail_idle_energy_j']:.1f} J of it early finishers idling)",
        f"facility peak {s['facility_peak_w']:.1f} W @ {dt_ms:.0f} ms    "
        f"floor {s['facility_floor_w']:.1f} W    "
        f"dynamic peak {s['dynamic_peak_w']:.1f} W",
        f"coincidence {s['coincidence_factor']:.3f}    "
        f"dynamic coincidence {s['dynamic_coincidence_factor']:.3f}    "
        f"peak/mean {s['peak_to_mean']:.2f}",
        f"balance: requests max/mean {s['requests_max_over_mean']:.2f}    "
        f"tokens max/mean {s['tokens_max_over_mean']:.2f}    "
        f"preemptions {s['preemptions']}",
        f"TTFT p99 {s['ttft_p99_s']:.3f} s    ITL p99 {s['itl_p99_ms']:.2f} ms    "
        f"energy/output token {s['energy_per_output_token_mj']:.1f} mJ",
    ]
    ax.text(0.0, 0.95, "\n".join(lines), fontsize=10, family="monospace",
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle(
        f"Fleet power trace -- {s['num_replicas']} replicas, routing = "
        f"{s['routing']}, {s['facility_peak_w']:.0f} W peak / "
        f"{s['avg_power_w']:.0f} W mean",
        fontsize=12, y=0.985)
    _stamp(fig, trace)
    return fig

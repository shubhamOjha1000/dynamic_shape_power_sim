# dynshape — dynamic `(batch, seq_len, mode)` for EnergAIzer

EnergAIzer predicts per-kernel latency, power and energy from a **traced** workload file — a
flat JSON list of kernels captured by running the real model once on a real GPU at one fixed
input shape. The shipped GPT-2 files are literally named

```
gpt2model_gpt2_pbf16_b8_s128_modeprefill.json
```

so **every shape you can evaluate is a shape somebody already traced.** Ask for `batch=13,
seq_len=377` and there is no file.

This repo makes that input shape **dynamic**. Trace three times, derive everything else.

```python
from dynshape import ShapeRewriter, RandomShapeGenerator, build_predictor, simulate
from dynshape.plot import plot_dashboard

rw    = ShapeRewriter.from_dir("templates/gpt2")     # learns from 3 anchor traces
reqs  = RandomShapeGenerator(seed=0).sample(40)      # random batch x seqlen x mode
pred  = build_predictor(pkg_path=PKG, lut_dir=LUT)   # EnergAIzer, or analytic fallback
trace = simulate(reqs, rw, pred)                     # time + average power per kernel
plot_dashboard(trace)
```

---

## How it works

Every numeric field in a traced entry is a monomial in the input dimensions:

```
value = const × B^a × S^b
```

Trace three times, moving **one input at a time**, and the exponents fall out by differencing:

```
T_base = trace(B=8,  S=128)
T_2B   = trace(B=16, S=128)     ← only batch changed   → recovers a
T_4S    = trace(B=8,  S=512)    ← only seqlen changed  → recovers b
```

```python
a = log(T_2B[i][field] / T_base[i][field], batch_ratio)
b = log(T_4S[i][field] / T_base[i][field], seq_ratio)
```

No architecture knowledge, no hard-coded kernel families, no per-model rules. A non-integer
exponent raises `ValueError` rather than rounding — if the power law does not hold for a field,
you want that loud, not baked silently into every downstream prediction.

### What the exponents turn out to be

| Field | a | b | Meaning |
|---|---|---|---|
| projection GEMM `dimM` = 1024 | 1 | 1 | tokens = B·S |
| projection GEMM `dimN` = 2304 | 0 | 0 | 3 × hidden — architecture constant |
| attention GEMM `batch` = 96 | **1** | **0** | B × n_heads |
| attention GEMM `dimM` = 128 | **0** | **1** | sequence position |
| softmax `dim` = 128 | 0 | 1 | reduces over the key axis |
| elementwise `dim` = 786432 | 1 | 1 | tokens × hidden |
| elementwise `dim` = 1572864 | 1 | **2** | B·heads·S·S — the score tensor |

That last row is why a single trace cannot be reverse-engineered by hand: `1572864` is both
`2 × 786432` and `B·h·S²`, and only differencing tells you which.

### `mode` is a third axis, not a rescaling

Prefill attention is a **square** — every token attends to every token. Decode attention is a
**strip** — one query row against the whole KV cache. No exponent turns a square into a strip:

```
prefill:  attn GEMM   M = S,  N = S            square
decode:   attn GEMM   M = 1,  N = context + 1  strip
```

**The `+1` is measured, not assumed.** The generated token's own K/V are appended to the cache
before attention runs, so it attends over one more key than the cache holds. Tracing GPT-2 decode
showed 48 of 242 entries differing from the original inferred rule — 12 blocks × 4 attention
entries, every one off by exactly one token:

| | inferred | measured |
|---|---|---|
| attention `dimN` | 128 | **129** |
| softmax `dim` | 128 | **129** |
| score tensor `dim` | 12288 | **12384** (= 96 × 129) |

That also makes decode's sequence axis **affine**, so the law is `const × B^a × (S + offset)^b`.
A pure power law in `S` fits with an exponent of 0.9958 rather than 1, and `learn_scaling` refuses
it — correctly. The shift is detected when the law is learned, never configured.

So the sequence exponent `b` is **split** into two independent axes — query length `Sq` and key
length `Sk` — giving `value = const × B^a × Sq^p × Sk^q` with `p + q = b`:

```
prefill(B, S)       →  Sq = S,  Sk = S
decode(B, context)  →  Sq = 1,  Sk = context
```

The split rule is eight lines in [`dynshape/template.py`](dynshape/template.py)
(`split_seq_exponent`) and is the **one hand-written piece** in the whole package.

---

## Validation

```
$ pytest -q
107 passed
```

**The headline test.** Learned from three anchors, the rewriter reproduces **all 25 shipped GPT-2
prefill templates exactly** — every field of every one of the 242 entries, across
`b ∈ {1,2,8,16,32} × s ∈ {128,512,1024,2048,4096}`. Not "close": `assert generated == reference`.

| Test file | What it pins down |
|---|---|
| `test_scaling.py` | the 25-file exact-match grid; every failure mode raises; two different anchor pairs learn identical rules |
| `test_decode.py` | decode's key axis is `context + 1`, measured; the shipped anchors give `decode_source == "measured"` with offset 1; the law reproduces a **held-out real trace** exactly; the query/key split still reduces to prefill when `Sq == Sk` |
| `test_workload.py` | reproducibility, isolated substreams, ranges, bucketing |
| `test_predictor.py` | `energy == power × time`; cache correctness and payoff; roofline direction |
| `test_simulate.py` | contiguous monotone timeline; energy conserved at all three levels; gaps at idle not zero; prefill hotter than decode; every figure renders |

### Decode can be measured instead of inferred — drop three files in

`ShapeRewriter` holds **two laws**, not one. Prefill always has its own, learned from three prefill
anchors. Decode gets a *second, independent* law as soon as three decode anchors exist — and when
they do, the inferred query/key split is never consulted:

```
prefill anchors (3)  ->  prefill law  ->  any prefill shape    verified 25/25
decode  anchors (3)  ->  decode  law  ->  any decode shape     verifiable the same way
```

Produce them with the artifact's own harness — three runs, minutes on any A100:

```bash
python run_model.py --model_type LanguageModel --model GPT2Model --precision bf16     --batch 8  --seqlen 128 --mode decode --trace --trace_save_to dec_b8_s128.csv
python run_model.py ... --batch 16 --seqlen 128 --mode decode ...   # -> batch exponent
python run_model.py ... --batch 8  --seqlen 512 --mode decode ...   # -> context exponent
python parse_trace.py --trace_path ... --parsed_save_to templates/gpt2/
```

**A ready-made notebook does all of this:**
[`notebooks/Trace_GPT2_Decode_Colab.ipynb`](notebooks/Trace_GPT2_Decode_Colab.ipynb). It clones the
artifact, reconstructs the missing `workload_config/` (which does not ship), traces four decode
shapes, and grades the inferred rule against the measurement. It opens with a **control** — retracing
a template that already exists and asserting an exact match — so a broken pipeline is caught on a
case with a known answer rather than misread as a decode finding. A free Colab CPU or T4 is enough;
tracing records op names and tensor shapes, not timings.

`from_dir` picks up any `..._modedecode.json` automatically. Check which law is in play:

```python
rw.decode_source        # 'inferred' today; 'measured' once the traces exist
```

If the real decode trace turns out to have a different number of kernels, that is **printed, not
swallowed** — a differing kernel list is precisely the thing worth discovering, since it would mean
the inferred split was wrong about more than exponents.

### What is **not** validated

**No decode ground truth ships with the artifact.** All 90 workload files are `modeprefill` —
`grep -c decode` returns 0 — even though the authors' own `run_gpt2.sh` has `MODE="prefill decode"`.

Tracing it (see the notebook below) settled three things that were previously assumptions:

| question | answer |
|---|---|
| does decode run the same 242 kernels in the same order? | **yes** — op sequence identical to prefill |
| was the inferred query/key split right? | **no** — off by one token on 48 of 242 entries, now corrected |
| does a decode law learned from 3 anchors generalise? | **yes** — reproduces a held-out 4th trace exactly |

**Those traces are now committed.** `templates/gpt2/` holds three real decode anchors
(`b8 s128`, `b16 s128`, `b8 s512`), so `decode_source` reports `measured` out of the box and
`split_seq_exponent` is never consulted for GPT-2.

A fourth real trace, `b16 s512`, is kept in [`tests/holdout/`](tests/holdout/) — deliberately
outside `templates/gpt2/`, where `from_dir` cannot reach it. It is never used to learn a law, only
to check one: `test_measured_decode_law_reproduces_the_holdout` fits on the three anchors and
predicts it exactly. A second test guards that separation, so the independent check cannot quietly
become a memorisation test.

Also assumed, and stated rather than tested: kernel costs are **additive** within a pass (no
overlap, no memory contention, no cache carry-over); the template is traced from **HuggingFace**,
so it has separate attention kernels where vLLM runs a fused paged one; and the tracer's keep-list
drops reshape / permute / transpose / view / cat / slice / embedding / dropout, treating data
movement as free.

---

## The two backends

| Backend | Needs | Use it for |
|---|---|---|
| `GeeBackend` | the EnergAIzer LUT database (separate ~500 MB download) | real numbers — 3.1–3.8% MAPE per kernel type |
| `AnalyticBackend` | nothing | making the pipeline, the tests and the graphs run with no LUT and no GPU |

`AnalyticBackend` is a roofline stand-in. It is labelled **SYNTHETIC** in its own `name`, in
`Trace.summary()`, and stamped on every figure it produces. It gets the *trends* right —
compute- vs memory-bound, batch scaling, precision, prefill vs decode — and the absolute numbers
roughly. Never quote one of its numbers as an EnergAIzer prediction.

`build_predictor()` picks the real one when the LUT is present and falls back loudly when it is
not, so a Colab session that skipped the download still produces graphs.

---

## Caching is mandatory, not an optimisation

A real lookup costs on the order of **50 ms**. One 32×2048 prefill expands to **242 kernels**;
a few hundred passes is ~100k lookups — hours. But the distinct *shapes* number in the low
hundreds, because the 12 transformer blocks are 12 literal repetitions of the same kernels and
batch sizes recur.

Measured on a 12-pass random stream: **2,904 kernels → 145 distinct shapes, 95% cache hit rate.**
Cache at *simulation* scope, not per pass.

---

## Decode dynamics: the KV cache grows, so the shape moves

A generation is not one decode call. It is a prefill, then N decode steps whose **KV cache grows by
one token each time** — so every step is a different attention shape:

```
prefill   b x 1024 tokens      attn 1024 x 1024    square
decode    ctx = 1024           attn    1 x 1024    strip
decode    ctx = 1025           attn    1 x 1025    longer strip
decode    ctx = 1026           attn    1 x 1026         ...
```

Attention work grows linearly across the generation while the projection GEMMs stay pinned at
`batch` rows. That asymmetry is why a long conversation gets slower and hotter the further it runs.

```python
from dynshape import generation, conversation

generation(batch=8, prompt_len=512, n_new_tokens=2000)   # one generation, step by step
conversation([(500, 100), (20, 100)], batch=1)           # multi-turn; cache carries across turns
```

Measured over 2,000 decode steps (analytic backend, batch 8, prompt 512):

| context | step time | avg power |
|---|---|---|
| 512 | 0.89 ms | 75 W |
| 1408 | 1.22 ms | 84 W |
| 2560 | 1.64 ms | 90 W |

**Context bucketing is what keeps this tractable.** At `context_bucket=128` those 2,000 steps
collapse to **94 distinct shapes with a 100% cache hit rate**; at `context_bucket=1` every step is
its own shape and the cache is useless. Rounding is *up*, matching Vidur, whose
`kv_cache_prediction_granularity` defaults to 64.

`conversation()` shows the multi-turn asymmetry directly — turn 2 prefills only its 20 new tokens
but its decode steps start from a 523-token cache: cheap to start, expensive to continue.

---

## L0 and L1: a real serving engine

Everything above prices a *shape*. Everything here decides **which shapes a real engine would
actually produce, and when** — and the answer is that prefill and decode happen in the *same*
forward pass.

```python
from dynshape import (TrafficConfig, generate_traffic, SchedulerConfig,
                      EngineConfig, run_engine, ShapeRewriter, build_predictor)
from dynshape.engine_plot import plot_engine_dashboard

reqs  = generate_traffic(TrafficConfig(interval="gamma", qps=6, cv=2.0,   # bursty
                                       length="zipf", theta=0.85,        # long-tailed
                                       num_requests=120, seed=0))
trace = run_engine(reqs, ShapeRewriter.from_dir("templates/gpt2"),
                   build_predictor(force_analytic=True), EngineConfig())
plot_engine_dashboard(trace)
```

| module | layer | what it decides | ported from |
|---|---|---|---|
| [`arrival.py`](dynshape/arrival.py) | L0 | when the next request arrives | Vidur — `static`, `poisson`, `gamma`, `trace` |
| [`lengths.py`](dynshape/lengths.py) | L0 | how big it is | Vidur — `fixed`, `uniform`, `zipf`, `trace` |
| [`traffic.py`](dynshape/traffic.py) | L0 | the two composed, with isolated RNGs | Vidur's `SyntheticRequestGenerator` |
| [`kvcache.py`](dynshape/kvcache.py) | L1 | how big the KV pool is, and who holds it | Vidur `MemoryPlanner` + block allocator |
| [`scheduler.py`](dynshape/scheduler.py) | L1 | who is in this batch | FSTS / vLLM V1 decode-first chunked prefill |
| [`mixed.py`](dynshape/mixed.py) | L2 | what kernels a **mixed** batch launches | new |
| [`engine.py`](dynshape/engine.py) | — | the run loop | new |

Multi-replica routing is deliberately out of scope: one replica.

### The scheduler, in two rules

```
decodes = [every running request whose prefill is done]   # protected, off the top
budget  = chunk_size - len(decodes)                       # what is left
prefill = fills the remainder, sliced to fit
```

Existing users are protected; new users are sacrificed. Because decodes come off the top, the
busier the system the less budget survives for prefill — at 1800 chatters a 4000-token prompt
takes 17 iterations instead of 2. For power it means every iteration carries **both**
compute-bound prefill and memory-bound decode instead of alternating between them: same energy,
lower peak, flatter ramp.

### What fuses in a mixed batch, and what cannot

The 242 template entries split into two classes, and the split is **derived from the learned
exponents** rather than hard-coded per kernel family:

| | rule | count | why |
|---|---|---|---|
| **fusible** | every field has `a == b`, so it depends only on `B·S` | **194** | a linear layer sees a bag of token rows and does not care which request each came from |
| **per request** | some field has `a ≠ b` | **48** | attention cares about nothing else |

Set `B=1, S=total_tokens` and the fusible entries are **exact, not approximated** — emitting one
GEMM at the summed token count is strictly more faithful than emitting one per request, and costs
nothing. The 48 are 12 blocks × 4 attention kernels: the same 48 the decode `+1` correction
touched, which is a useful independent confirmation of both.

Attention cannot be fused this way. vLLM issues one ragged `flash_attn_varlen_func` launch whose
shape is a pair of offset arrays; EnergAIzer's LUT is keyed on rectangles and has no varlen entry.
Under **eager** attention there is nothing to lose, which is what makes this the self-consistent
choice for v1.

### Preemption is the interesting event

The KV accounting comes from Vidur rather than FSTS for one reason: **FSTS cannot preempt.** When
the pool fills, Vidur picks a victim — newest arrival first, so requests closest to finishing keep
their cache — throws away its KV, and re-prefills everything it had generated. Real GPU work, real
watts, no new output, and *nothing in the arrival pattern predicts it*.

GPT-2 never hits this on an A100: its cache costs 36 KiB/token, so the pool holds about a million
tokens. Set `SchedulerConfig(num_blocks=...)` explicitly to shrink it and study the behaviour a
70B model would reach naturally.

### Reporting power the way a meter would

Per-iteration events are the natural output of a kernel simulator and the wrong thing to hand
anyone: the steps have variable width, so the series cannot be lined up against an NVML capture,
averaged across runs, or used to quote a peak.

```python
t_ms, watts = trace.resample(dt_ms=1.0)                      # fixed grid, energy-conserving
t_ms, watts = trace.resample(dt_ms=1.0, smooth_tau_ms=5.0)   # through a board response
```

`resample()` is a box filter — the integral of the returned series equals `total_energy_j` at any
sample rate — and `smooth_tau_ms` applies a one-pole board response, because a real sensor never
sees the square edges a kernel trace produces. **A peak is not a number until you say over what
window**: the same trace peaks differently at a 0.1 ms aperture than at 100 ms, while the mean and
the integral are invariant.

Each iteration also carries a **work vector** — FLOPs and bytes implied by the shapes, computed in
[`work.py`](dynshape/work.py) with no power model involved. It is what separates the two error
sources: if a predicted trace disagrees with a measured one, the work vector says whether the
simulator got the *work* wrong or the *conversion to watts* wrong. With only watts, those are
indistinguishable — and with a SYNTHETIC predictor in the loop, that distinction is the whole game.

FLOPs split between prefill and decode **exactly** (a fused GEMM's rows each belong to one request,
and FLOPs are linear in the row count). Bytes do not — the weight matrix is read once for the whole
batch, which is the entire point of fusing it — so the split is applied to the first and refused for
the second.

### Three kinds of time

`KERNEL`, `GAP` (one per iteration, at idle) and `IDLE` (waiting for an arrival). The third only
exists once there is a real arrival process, and it is why `avg_power_w_wallclock` and
`avg_power_w_busy` are reported separately — at low load a report that quietly averages only the
busy segments overstates facility draw substantially.

---

## Output

Per **kernel**: `t_start_ms, time_ms, power_W, energy_J, op, shape`
Per **pass**: `time_ms, avg_power_W, energy_J, tokens_per_s, energy_per_token_mJ`

Three graphs, because they answer different questions:

| Figure | Question |
|---|---|
| `plot_power_timeline` | *when* is the machine hot — power vs wall clock, a staircase |
| `plot_request_scatter` | *which shapes* are hot — time vs average power, one point per pass |
| `plot_shape_sweep` | *how* power moves with each dynamic input — lines over a deterministic grid |

`plot_dashboard` puts them on one figure.

---

## Running it

**Colab (recommended, CPU is fine).** Two notebooks, both clone this repo from GitHub and run the
test suite before anything else:

| notebook | what it shows |
|---|---|
| [`Serving_Engine_Power_Sim_Colab.ipynb`](notebooks/Serving_Engine_Power_Sim_Colab.ipynb) | L0 + L1 + mixed batching — arrivals, lengths, the scheduler, preemption, fused vs concatenated |
| [`Dynamic_Shape_Power_Sim_Colab.ipynb`](notebooks/Dynamic_Shape_Power_Sim_Colab.ipynb) | the shape rewriter on its own — any `(batch, seq_len, mode)`, no engine |

A third, [`Trace_GPT2_Decode_Colab.ipynb`](notebooks/Trace_GPT2_Decode_Colab.ipynb), regenerates
the measured decode templates from scratch.

**Locally:**

```bash
pip install -r requirements.txt
pytest -q
python run_demo.py --n 40 --seed 0            # analytic fallback
python run_demo.py --sweep                    # deterministic grid
python run_demo.py --pkg-path /path/to/energaizer-ispass26-artifact-main \
                   --lut-dir  /path/to/database/data     # real EnergAIzer
```

---

## Layout

```
dynshape/
  template.py     learn_scaling, rewrite_dims, the query/key split   ← the core
  workload.py     random (batch, seq_len, mode) stream + grid sweep
  predictor.py    GeeBackend | AnalyticBackend + the cache
  simulate.py     timeline assembly → per-kernel and per-pass records
  plot.py         the three shape-stream graphs
  work.py         FLOPs and bytes from a shape -- no backend, no LUT

  arrival.py      L0 — static | poisson | gamma | trace, + synthetic λ(t)
  lengths.py      L0 — fixed | uniform | zipf | trace
  traffic.py      L0 — the two composed, isolated RNGs per stream
  entities.py     SimRequest, Batch, Piece — the request lifecycle
  kvcache.py      L1 — MemoryPlanner + block allocator + watermark
  scheduler.py    L1 — decode-first chunked prefill, preemption/restart
  mixed.py        L2 — a mixed batch → kernels; what fuses and what cannot
  engine.py       the run loop; KERNEL / GAP / IDLE
  engine_plot.py  the six-panel engine dashboard

templates/gpt2/   25 shipped prefill traces + 3 measured decode anchors
tests/holdout/    a 4th real decode trace, deliberately unreachable as an anchor
notebooks/        the shape demo, the decode tracer, the serving-engine notebook
run_demo.py       CLI
```

## Credit

Built on **EnergAIzer** (ISPASS'26) — the kernel-level latency/power/energy model this package
feeds shapes into. The traced GPT-2 templates in `templates/gpt2/` come from its artifact
(MIT-licensed).

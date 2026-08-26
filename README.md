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
prefill:  attn GEMM   M = S,  N = S        square
decode:   attn GEMM   M = 1,  N = context  strip
```

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
| `test_decode.py` | the query/key split **reduces to prefill exactly** when `Sq == Sk`, which ties the unvalidated decode path to the validated prefill one; decode attention is a strip; softmax and mask shapes agree with what attention produced |
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

**Decode has no ground truth anywhere.** All 90 workload files shipped with the EnergAIzer
artifact are `modeprefill` — `grep -c decode` returns 0. The decode rule is derived from what
`run_model.get_input()` actually feeds the model (one token per sequence plus a length-`seqlen`
KV cache) and is checked for *internal* consistency only. Treat decode numbers as structurally
sound and empirically unconfirmed.

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

## What this deliberately is not

There is **no L0** (arrival process) and **no L1** (scheduler) here. No Poisson arrivals, no KV
admission, no chunked prefill, no preemption. Those layers decide *which* shapes a real engine
would produce; [`dynshape/workload.py`](dynshape/workload.py) just emits a varied spread so the
dynamic-shape machinery has something to chew on.

Swapping that one file for a real Vidur timeline **is** the entire L0/L1 integration — everything
downstream of it stays exactly as written.

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

**Colab (recommended, CPU is fine):** open
[`notebooks/Dynamic_Shape_Power_Sim_Colab.ipynb`](notebooks/Dynamic_Shape_Power_Sim_Colab.ipynb).
It clones this repo and the EnergAIzer framework from GitHub, optionally downloads the LUT, runs
the tests, and draws the graphs. Nothing runs locally.

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
  template.py    learn_scaling, rewrite_dims, the query/key split   ← the core
  workload.py    random (batch, seq_len, mode) stream + grid sweep
  predictor.py   GeeBackend | AnalyticBackend + the cache
  simulate.py    timeline assembly → per-kernel and per-pass records
  plot.py        the three graphs
templates/gpt2/  the 25 shipped GPT-2 traces (3 anchors + 22 test targets)
tests/           107 tests
run_demo.py      CLI
```

## Credit

Built on **EnergAIzer** (ISPASS'26) — the kernel-level latency/power/energy model this package
feeds shapes into. The traced GPT-2 templates in `templates/gpt2/` come from its artifact
(MIT-licensed).

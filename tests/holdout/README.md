# Held-out decode trace — never an anchor

`gpt2model_gpt2_pbf16_b16_s512_modedecode.json` is a real GPT-2 decode trace, captured by
[`notebooks/Trace_GPT2_Decode_Colab.ipynb`](../../notebooks/Trace_GPT2_Decode_Colab.ipynb) in the
same run as the three anchors in `templates/gpt2/`.

**It lives here, not there, on purpose.** `ShapeRewriter.from_dir` only ever reads
`templates/gpt2/`, so this trace cannot be picked up as an anchor — it is never used to learn a
law. That keeps `test_measured_decode_law_reproduces_the_holdout` an *independent* check: the law
is fitted on `b8 s128`, `b16 s128` and `b8 s512`, and then asked to predict a shape it has never
seen.

Moving this file into `templates/gpt2/` would silently convert the only independent decode
validation into a memorisation test.

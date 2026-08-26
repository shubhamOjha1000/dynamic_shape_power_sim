import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TEMPLATE_DIR = os.path.join(ROOT, "templates", "gpt2")

#: A real decode trace kept OUT of TEMPLATE_DIR so it can never become an anchor.
#: See tests/holdout/README.md.
HOLDOUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holdout")

B0, S0 = 8, 128
#: Every (batch, seqlen) prefill template that ships with the EnergAIzer artifact.
SHIPPED_BATCHES = [1, 2, 8, 16, 32]
SHIPPED_SEQLENS = [128, 512, 1024, 2048, 4096]


def template_path(b, s):
    return os.path.join(TEMPLATE_DIR, f"gpt2model_gpt2_pbf16_b{b}_s{s}_modeprefill.json")


@pytest.fixture(scope="session")
def rewriter():
    from dynshape import ShapeRewriter
    return ShapeRewriter.from_dir(TEMPLATE_DIR, b0=B0, s0=S0)


@pytest.fixture(scope="session")
def predictor():
    from dynshape import build_predictor
    return build_predictor(force_analytic=True)

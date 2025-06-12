#!/usr/bin/env python
"""Minimal RecurrentGemma sampler using Preset to auto-fill config."""

from pathlib import Path
import sentencepiece as spm
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg    # alias for the JAX API
import jax

# ── 1. point at your local files ──────────────────────────────────────────
CKPT_DIR = Path("2b").resolve()              # contains _METADATA, checkpoint/, etc.
TOK_FILE = Path("tokenizer.model").resolve()  # SentencePiece vocab

# ── 2. restore the weights PyTree (OCDBT) ─────────────────────────────────
restored = ocp.PyTreeCheckpointer().restore(str(CKPT_DIR))
params   = restored.get("params", restored)   # unwrap TrainState if present

# ── 3. pick the right Preset & build config from checkpoint + preset ──────
#    (this injects the correct window_size, layer counts, hidden_size, etc.)
preset = rg.Preset.RECURRENT_GEMMA_2B_V1      # or _9B_V1 for the 9B variant :contentReference[oaicite:0]{index=0}
cfg    = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

# ── 4. instantiate model + tokenizer + sampler ────────────────────────────
model   = rg.Griffin(cfg)
vocab   = spm.SentencePieceProcessor(model_file=str(TOK_FILE))
sampler = rg.Sampler(
    model=model,
    vocab=vocab,
    params=params,
    deterministic_sampling=True           # greedy by default
)

# ── 5. generate a sentence ────────────────────────────────────────────────
out = sampler(
    ["One sentence about TPUs."],
    total_generation_steps=64,
)
print(out.text[0])

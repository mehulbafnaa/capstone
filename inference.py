#!/usr/bin/env python
"""Minimal RecurrentGemma sampler using Preset to auto-fill config."""

from pathlib import Path
import sentencepiece as spm
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg    # alias for the JAX API
import jax

# ── 1. point at your local files ──────────────────────────────────────────
CKPT_DIR = Path("2b-it/2b-it").resolve()              # contains _METADATA, checkpoint/, etc.
TOK_FILE = Path("2b-it/tokenizer.model").resolve()  # SentencePiece vocab

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

# # ── 5. generate a sentence ────────────────────────────────────────────────
# out = sampler(
#     ["One sentence about TPUs."],
#     total_generation_steps=200,
# )


out = sampler(
    ["You are a formal theorem prover working with Lean 4. Your task is to complete the proof for the given theorem.\n\n## Example Pattern\nHere's how to approach Lean proofs:\n\n```lean\ntheorem example_add_zero (n : ℕ) : n + 0 = n := by\n  rfl  -- This works because n + 0 = n by definition\n```\n\n```lean\ntheorem example_succ (n : ℕ) : n + 1 = Nat.succ n := by\n  rfl  -- This works because + 1 is defined as Nat.succ\n```\n\n## Available Tactics\n- `rfl`: Use when goal is true by definition/reflexivity\n- `simp`: Simplify using standard lemmas\n- `rw [lemma_name]`: Rewrite using a specific lemma\n- `exact lemma_name`: Apply exact lemma that matches goal\n\n## Standard Library Lemmas\n- `Nat.add_comm : ∀ a b : ℕ, a + b = b + a`\n- `Nat.add_assoc : ∀ a b c : ℕ, (a + b) + c = a + (b + c)`\n- `Nat.zero_add : ∀ n : ℕ, 0 + n = n`\n- `Nat.add_zero : ∀ n : ℕ, n + 0 = n`\n\n## Your Task\nComplete this theorem by replacing `sorry` with the correct proof:\n\n```lean\ntheorem add_comm_simple (a b : ℕ) : a + b = b + a := by\n  sorry\n```\n\n## Instructions\n1. Look at the goal: `a + b = b + a`\n2. This is commutativity of addition\n3. Check if there's a standard library lemma that directly proves this\n4. Replace `sorry` with the appropriate tactic\n\n## Expected Format\nProvide only the complete theorem with the proof filled in:\n\n```lean\ntheorem add_comm_simple (a b : ℕ) : a + b = b + a := by\n  [your proof here]\n```\n\nYour proof:"],
    total_generation_steps=1000,
)

print(out.text[0])

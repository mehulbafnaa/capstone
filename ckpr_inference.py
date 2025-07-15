

#!/usr/bin/env python
"""
Minimal RecurrentGemma sampler that loads a fine-tuned checkpoint.
"""

from pathlib import Path
import sentencepiece as spm
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import jax

# ── 1. Point to your local file paths ───────────────────────────────────────
# Point to the TOP-LEVEL directory managed by Orbax, not a specific step.
CKPT_MANAGER_DIR = Path("finetuning_checkpoints/").resolve()

TOK_FILE = Path("2b/tokenizer.model").resolve()

# ── 2. Restore the fine-tuned weights using CheckpointManager ───────────────
# Initialize the manager for the directory.
manager = ocp.CheckpointManager(CKPT_MANAGER_DIR)

# Find the latest saved step (e.g., 117).
latest_step = manager.latest_step()
if latest_step is None:
    raise FileNotFoundError(f"No checkpoints found in {CKPT_MANAGER_DIR}")

print(f"Loading fine-tuned checkpoint from step: {latest_step}")

# Use PyTreeCheckpointer with the full path to the saved item.
item_path = manager.directory / str(latest_step) / 'default'
restored_state = ocp.PyTreeCheckpointer().restore(str(item_path))


# Extract the 'params' PyTree from the restored TrainState.
params = restored_state.get("params", restored_state)

# CRITICAL FIX: Explicitly gather the sharded parameters onto a single device.
# This converts the distributed arrays (GlobalDeviceArray) from the checkpoint
# into regular arrays, which prevents the "Mosaic kernels cannot be automatically
# partitioned" error during inference.
print("Gathering sharded parameters to a single device...")
params = jax.device_get(params)


# ── 3. Pick the right Preset & build config from checkpoint + preset ───────
# This part is the same as before. The config is derived from the loaded params.
preset = rg.Preset.RECURRENT_GEMMA_2B_V1
cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

# ── 4. Instantiate model + tokenizer + sampler ─────────────────────────────
model = rg.Griffin(cfg)
vocab = spm.SentencePieceProcessor(model_file=str(TOK_FILE))
# MODIFICATION: Reverting to deterministic sampling as per the reference script.
sampler = rg.Sampler(
    model=model,
    vocab=vocab,
    params=params,
    deterministic_sampling=True,  # Use greedy sampling by default.
)

# ── 5. Generate text using your theorem-proving prompt ─────────────────────
# Using the detailed prompt you provided.
prompt = """You are a formal theorem prover working with Lean 4. Your task is to complete the proof for the given theorem.

## Example Pattern
Here's how to approach Lean proofs:

```lean
theorem example_add_zero (n : ℕ) : n + 0 = n := by
  rfl  -- This works because n + 0 = n by definition
```lean
theorem example_succ (n : ℕ) : n + 1 = Nat.succ n := by
  rfl  -- This works because + 1 is defined as Nat.succ
```

## Available Tactics
- `rfl`: Use when goal is true by definition/reflexivity
- `simp`: Simplify using standard lemmas
- `rw [lemma_name]`: Rewrite using a specific lemma
- `exact lemma_name`: Apply exact lemma that matches goal

## Standard Library Lemmas
- `Nat.add_comm : ∀ a b : ℕ, a + b = b + a`
- `Nat.add_assoc : ∀ a b c : ℕ, (a + b) + c = a + (b + c)`
- `Nat.zero_add : ∀ n : ℕ, 0 + n = n`
- `Nat.add_zero : ∀ n : ℕ, n + 0 = n`

## Your Task
Complete this theorem by replacing `sorry` with the correct proof:

```lean
theorem add_comm_simple (a b : ℕ) : a + b = b + a := by
  sorry
```

## Instructions
1. Look at the goal: `a + b = b + a`
2. This is commutativity of addition
3. Check if there's a standard library lemma that directly proves this
4. Replace `sorry` with the appropriate tactic

## Expected Format
Provide only the complete theorem with the proof filled in:

```lean
theorem add_comm_simple (a b : ℕ) : a + b = b + a := by
  [your proof here]
```

Your proof:"""

print("Generating response...")
# MODIFICATION: Removed top_k as it's not used with deterministic sampling.
out = sampler(
    [prompt],
    total_generation_steps=100,
)

print("\n--- Generated Output ---")
print(out.text[0])

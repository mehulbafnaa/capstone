#!/usr/bin/env python
"""
Interactive RecurrentGemma REPL — loads checkpoint once, then loops on prompts.
"""

from pathlib import Path
import sentencepiece as spm
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import jax

# 1. Paths to your local checkpoint and vocab
CKPT_DIR = Path("2b/2b").resolve()              # contains _METADATA, checkpoint/, etc.
TOK_FILE = Path("2b/tokenizer.model").resolve()  # SentencePiece vocab

# 2. Restore OCDBT weights
restored = ocp.PyTreeCheckpointer().restore(str(CKPT_DIR))
params   = restored.get("params", restored)

# 3. Auto-fill config via Preset
cfg = rg.GriffinConfig.from_flax_params_or_variables(
    params, preset=rg.Preset.RECURRENT_GEMMA_2B_V1
)

# 4. Build model, tokenizer, sampler
model   = rg.Griffin(cfg)
vocab   = spm.SentencePieceProcessor(model_file=str(TOK_FILE))
sampler = rg.Sampler(model=model, vocab=vocab, params=params, deterministic_sampling=True, is_it_model=True)

# 5. Interactive loop
print("RecurrentGemma REPL ready. Type 'exit' or Ctrl-D to quit.")
while True:
    try:
        prompt = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt or prompt.lower() in ("exit", "quit"):
        break
    out = sampler([prompt], total_generation_steps=10000)
    print(out.text[0])
print("Goodbye!")

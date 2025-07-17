

# from pathlib import Path
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# from recurrentgemma.jax import Griffin as CheckpointedGriffin
# import sentencepiece as spm
# from typing import Any, Tuple
# import jax
# import jax.numpy as jnp

# def load_recurrent_gemma_model(
#     ckpt_dir: Path,
#     tok_file: Path,
#     params_dtype: Any = jnp.float32,
#     use_checkpointing: bool = False
# ) -> Tuple[rg.Griffin, spm.SentencePieceProcessor, Any, rg.Sampler]:
#     """
#     Loads the RecurrentGemma model, with an option for gradient checkpointing.
#     """
#     print(f"Loading model from: {ckpt_dir}")
#     print(f"Loading tokenizer from: {tok_file}")

#     # 1. Load the tokenizer to get the one, true vocab size.
#     vocab = spm.SentencePieceProcessor(model_file=str(tok_file))

#     # 2. Get the base configuration from the preset for all correct hyperparameters.
#     preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#     base_cfg = rg.GriffinConfig.from_preset(preset)

#     # 3. Manually create the final, correct config by replacing the one wrong value.
#     # This bypasses the broken inference function.
#     cfg = base_cfg._replace(vocab_size=vocab.vocab_size())

#     # 4. Load the raw parameters using the standard Orbax checkpointer.
#     restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
#     params = restored.get("params", restored)

#     if params_dtype is not None:
#         params = jax.tree.map(lambda x: x.astype(params_dtype), params)

#     # 5. Create the model instance with our manually constructed, foolproof config.
#     if use_checkpointing:
#         print("Using CheckpointedGriffin model for gradient checkpointing.")
#         model_cls = CheckpointedGriffin
#     else:
#         model_cls = rg.Griffin

#     model = model_cls(cfg)

#     # 6. Create the sampler and return.
#     sampler = rg.Sampler(
#         model=model,
#         vocab=vocab,
#         params=params,
#         deterministic_sampling=True,
#         is_it_model=False
#     )
#     return model, vocab, params, sampler



#!/usr/bin/env python3

from pathlib import Path
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
from recurrentgemma.jax import Griffin as CheckpointedGriffin
import sentencepiece as spm
from typing import Any, Tuple
import jax
import jax.numpy as jnp
import dataclasses
from finetuning.config import CKPT_DIR, TOK_FILE

def load_recurrent_gemma_model(
    ckpt_dir: Path,
    tok_file: Path,
    params_dtype: Any = jnp.float32,
    use_checkpointing: bool = False
) -> Tuple[rg.Griffin, spm.SentencePieceProcessor, Any, rg.Sampler]:
    """
    Loads the RecurrentGemma model, with an option for gradient checkpointing.
    """
    print(f"Loading model from: {ckpt_dir}")
    print(f"Loading tokenizer from: {tok_file}")

    # 1. Load the tokenizer to get the one, true vocab size.
    vocab = spm.SentencePieceProcessor(model_file=str(tok_file))

    # 2. Get the base configuration from the preset for all correct hyperparameters.
    preset = rg.Preset.RECURRENT_GEMMA_2B_V1
    base_cfg = rg.GriffinConfig.from_preset(preset)

    # 3. Manually create the final, correct config by replacing the one wrong value.
    # This bypasses the broken inference function.
    cfg = base_cfg._replace(vocab_size=vocab.vocab_size())

    # 4. Load the raw parameters using the standard Orbax checkpointer.
    restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    params = restored.get("params", restored)

    if params_dtype is not None:
        params = jax.tree.map(lambda x: x.astype(params_dtype), params)

    # 5. Create the model instance with our manually constructed, foolproof config.
    if use_checkpointing:
        print("Using CheckpointedGriffin model for gradient checkpointing.")
        model_cls = CheckpointedGriffin
    else:
        model_cls = rg.Griffin

    model = model_cls(cfg)

    # 6. Create the sampler and return.
    sampler = rg.Sampler(
        model=model,
        vocab=vocab,
        params=params,
        deterministic_sampling=True,
        is_it_model=False
    )
    
    # --- ADDED PRINT STATEMENT ---
    print(f"✅ Successfully created model with final vocab_size: {model.config.vocab_size}")
    
    return model, vocab, params, sampler

# --- ADDED MAIN FUNCTIONALITY ---
if __name__ == "__main__":


    print("--- Running Model Loader Standalone ---")
    try:
        model, vocab, params, sampler = load_recurrent_gemma_model(
            ckpt_dir=CKPT_DIR,
            tok_file=TOK_FILE,
        )
        print("\n---  Model loading successful! ---")
        print(f"Model Type: {type(model)}")
        print(f"Tokenizer Vocab Size: {vocab.vocab_size()}")
        print(f"Number of parameter arrays: {len(jax.tree_util.tree_leaves(params))}")

    except FileNotFoundError as e:
        print(f"\n---  ERROR: Could not load model. ---")
        print(f"Details: {e}")
        print("Please ensure the CKPT_DIR and TOK_FILE paths are correct in the script.")
    except Exception as e:
        print(f"\n---  An unexpected error occurred: ---")
        print(e)


# from pathlib import Path
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# from recurrentgemma.jax import Griffin as CheckpointedGriffin
# import sentencepiece as spm
# from typing import Any, Tuple
# import jax
# import jax.numpy as jnp
# import flax.linen as nn

# def load_recurrent_gemma_model(
#     ckpt_dir: Path,
#     tok_file: Path,
#     params_dtype: Any = jnp.float32,
#     use_checkpointing: bool = False
# ) -> Tuple[rg.Griffin, spm.SentencePieceProcessor, Any, rg.Sampler]:
#     """
#     Loads the RecurrentGemma model, with an option for gradient checkpointing.
#     """
#     if not ckpt_dir.exists():
#         raise FileNotFoundError(f"Checkpoint directory not found at: {ckpt_dir}")
#     if not tok_file.exists():
#         raise FileNotFoundError(f"Tokenizer file not found at: {tok_file}")

#     print(f"Loading model from: {ckpt_dir}")
#     print(f"Loading tokenizer from: {tok_file}")

#     restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
#     params = restored.get("params", restored)

#     if params_dtype is not None:
#         params = jax.tree.map(lambda x: x.astype(params_dtype), params)

#     preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#     cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

#     # Select the model class based on the checkpointing flag
#     if use_checkpointing:
#         print("Using CheckpointedGriffin model for gradient checkpointing.")
#         model_cls = CheckpointedGriffin
#     else:
#         model_cls = rg.Griffin

#     model = model_cls(cfg)

#     vocab = spm.SentencePieceProcessor(model_file=str(tok_file))
#     sampler = rg.Sampler(
#         model=model,
#         vocab=vocab,
#         params=params,
#         deterministic_sampling=True,
#         is_it_model=False
#     )
#     return model, vocab, params, sampler




from pathlib import Path
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
from recurrentgemma.jax import Griffin as CheckpointedGriffin
import sentencepiece as spm
from typing import Any, Tuple
import jax
import jax.numpy as jnp

def load_recurrent_gemma_model(
    ckpt_dir: Path,
    tok_file: Path,
    params_dtype: Any = jnp.float32,
    use_checkpointing: bool = False
) -> Tuple[rg.Griffin, spm.SentencePieceProcessor, Any, rg.Sampler]:
    """
    Loads the RecurrentGemma model, with an option for gradient checkpointing.
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found at: {ckpt_dir}")
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found at: {tok_file}")

    print(f"Loading model from: {ckpt_dir}")
    print(f"Loading tokenizer from: {tok_file}")

    # 1. Load tokenizer FIRST to get the correct vocabulary size
    vocab = spm.SentencePieceProcessor(model_file=str(tok_file))

    # 2. Load the checkpoint parameters
    restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    params = restored.get("params", restored)

    if params_dtype is not None:
        params = jax.tree.map(lambda x: x.astype(params_dtype), params)

    # 3. Get the base model config from the preset
    preset = rg.Preset.RECURRENT_GEMMA_2B_V1
    base_cfg = rg.GriffinConfig.from_preset(preset)

    # 4. CRITICAL: Create a NEW config with the vocab_size replaced
    cfg = base_cfg._replace(vocab_size=vocab.vocab_size())

    # Select the model class based on the checkpointing flag
    if use_checkpointing:
        print("Using CheckpointedGriffin model for gradient checkpointing.")
        model_cls = CheckpointedGriffin
    else:
        model_cls = rg.Griffin

    model = model_cls(cfg)

    sampler = rg.Sampler(
        model=model,
        vocab=vocab,
        params=params,
        deterministic_sampling=True,
        is_it_model=False
    )
    return model, vocab, params, sampler
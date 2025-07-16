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
#     params_dtype: Any = jnp.float32 # Add params_dtype argument with a default
# ) -> Tuple[rg.Griffin, spm.SentencePieceProcessor, Any, rg.Sampler]:
#     """
#     Loads the RecurrentGemma model, tokenizer, parameters, and sampler.

#     Args:
#         ckpt_dir: Path to the model checkpoint directory.
#         tok_file: Path to the SentencePieceProcessor tokenizer model file.
#         params_dtype: Optional. The desired dtype for the model parameters. Defaults to jnp.float32.

#     Returns:
#         A tuple containing:
#         - model: The RecurrentGemma Griffin model instance.
#         - vocab: The SentencePieceProcessor tokenizer instance.
#         - params: The loaded model parameters.
#         - sampler: The RecurrentGemma Sampler instance.
#     """
#     if not ckpt_dir.exists():
#         raise FileNotFoundError(f"Checkpoint directory not found at: {ckpt_dir}")
#     if not tok_file.exists():
#         raise FileNotFoundError(f"Tokenizer file not found at: {tok_file}")

#     print(f"Loading model from: {ckpt_dir}")
#     print(f"Loading tokenizer from: {tok_file}")

#     restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
#     params = restored.get("params", restored)

#     # Cast parameters to the desired dtype if specified
#     if params_dtype is not None:
#         params = jax.tree.map(lambda x: x.astype(params_dtype), params)

#     # Auto-fill config via Preset
#     preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#     cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

#     model = rg.Griffin(cfg)
#     vocab = spm.SentencePieceProcessor(model_file=str(tok_file))
#     sampler = rg.Sampler(
#         model=model,
#         vocab=vocab,
#         params=params,
#         deterministic_sampling=True, # Use deterministic sampling for consistency
#         is_it_model=True # Assuming it's an instruction-tuned model
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
import flax.linen as nn

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

    restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    params = restored.get("params", restored)

    if params_dtype is not None:
        params = jax.tree.map(lambda x: x.astype(params_dtype), params)

    preset = rg.Preset.RECURRENT_GEMMA_2B_V1
    cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

    # Select the model class based on the checkpointing flag
    if use_checkpointing:
        print("Using CheckpointedGriffin model for gradient checkpointing.")
        model_cls = CheckpointedGriffin
    else:
        model_cls = rg.Griffin

    model = model_cls(cfg)

    vocab = spm.SentencePieceProcessor(model_file=str(tok_file))
    sampler = rg.Sampler(
        model=model,
        vocab=vocab,
        params=params,
        deterministic_sampling=True,
        is_it_model=False
    )
    return model, vocab, params, sampler
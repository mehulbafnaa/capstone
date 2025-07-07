from pathlib import Path
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
from recurrentgemma.jax.griffin import Block
import sentencepiece as spm
from typing import Any, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn

class CheckpointedGriffin(rg.Griffin):
    @nn.compact
    def __call__(
        self, 
        tokens,
        segment_pos,
        cache=None,
        return_full_cache=False,
    ):
        # During training, we should not be using the cache.
        assert cache is None, 'Cache should not be used during training.'

        # Embed the tokens.
        x = self.embedder(tokens)

        # Normalize the embeddings.
        x = self.embedder_norm(x)

        # Run the blocks, applying gradient checkpointing to each one.
        for i in range(self.config.num_layers):
            x, _ = nn.remat(Block)(self.config, self.block_types)(x, segment_pos, None)

        # Apply the final normalization layer.
        x = self.final_norm(x)

        # Compute the logits.
        logits = self.embedder.decode(x)

        return logits, None

def load_recurrent_gemma_model(
    ckpt_dir: Path,
    tok_file: Path,
    params_dtype: Any = jnp.float32 # Add params_dtype argument with a default
) -> Tuple[rg.Griffin, spm.SentencePieceProcessor, Any, rg.Sampler]:
    """
    Loads the RecurrentGemma model, tokenizer, parameters, and sampler.

    Args:
        ckpt_dir: Path to the model checkpoint directory.
        tok_file: Path to the SentencePieceProcessor tokenizer model file.
        params_dtype: Optional. The desired dtype for the model parameters. Defaults to jnp.float32.

    Returns:
        A tuple containing:
        - model: The RecurrentGemma Griffin model instance.
        - vocab: The SentencePieceProcessor tokenizer instance.
        - params: The loaded model parameters.
        - sampler: The RecurrentGemma Sampler instance.
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found at: {ckpt_dir}")
    if not tok_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found at: {tok_file}")

    print(f"Loading model from: {ckpt_dir}")
    print(f"Loading tokenizer from: {tok_file}")

    restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
    params = restored.get("params", restored)

    # Cast parameters to the desired dtype if specified
    if params_dtype is not None:
        params = jax.tree.map(lambda x: x.astype(params_dtype), params)

    # Auto-fill config via Preset
    preset = rg.Preset.RECURRENT_GEMMA_2B_V1
    cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

    model = CheckpointedGriffin(cfg)
    vocab = spm.SentencePieceProcessor(model_file=str(tok_file))
    sampler = rg.Sampler(
        model=model,
        vocab=vocab,
        params=params,
        deterministic_sampling=True, # Use deterministic sampling for consistency
        is_it_model=True # Assuming it's an instruction-tuned model
    )
    return model, vocab, params, sampler

# finetuning/sharding.py

import jax
import jax.tree_util as jtu
from jax.sharding import PartitionSpec
import re

def get_recurrent_gemma_sharding_spec(pytree):
    """
    A definitive, architecture-informed FSDP sharding strategy for Recurrent Gemma.
    """
    # Rules are applied in order; the first match wins.
    sharding_rules = [
        # 1. Embeddings: Shard the vocabulary axis.
        (r"embedder/input_embedding", PartitionSpec("fsdp", None)),

        # 2. MLP Blocks: Handle the 3D and 2D weights separately.
        (r"blocks\.\d+\.mlp_block\.ffw_up\.w", PartitionSpec(None, None, "fsdp")),
        (r"blocks\.\d+\.mlp_block\.ffw_down\.kernel", PartitionSpec("fsdp", None)),

        # 3. Attention Blocks: Shard all kernel matrices.
        (r"blocks\.\d+\.attention_block\..*\.kernel", PartitionSpec("fsdp", None)),

        # 4. Recurrent Blocks: Only shard the main linear layer kernels.
        (r"blocks\.\d+\.recurrent_block\.linear_.*\.kernel", PartitionSpec("fsdp", None)),

        # 5. Default Rule: Replicate EVERYTHING else.
        (r".*", PartitionSpec()),
    ]

    def spec_fn(path, leaf):
        """Finds the first matching regex rule for a given parameter path."""
        path_str = jtu.keystr(path)
        for pattern, spec in sharding_rules:
            if re.fullmatch(pattern, path_str):
                return spec
        raise ValueError(f"No sharding rule found for parameter: {path_str}")

    return jtu.tree_map_with_path(spec_fn, pytree)
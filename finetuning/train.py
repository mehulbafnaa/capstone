

# #!/usr/bin/env python3
# """
# Fool-proof training script for Recurrent-Gemma-2B on TPU v4-16
# Full-parameter sharding (FSDP) with safe fall-back
# """

# # 0) Force TPU and suppress CUDA noise
# import os
# os.environ["JAX_PLATFORMS"] = "tpu"
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# # 1) Core imports
# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from flax.struct import field
# from tqdm import tqdm
# from pathlib import Path
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental.pjit import pjit
# import jax.tree_util as jtu
# import tensorflow_datasets as tfds
# from jax.experimental import multihost_utils
# from jax.experimental.shard_map import shard_map   # <-- NEW

# from utils.model_loader import load_recurrent_gemma_model
# from finetuning.data_pipeline import get_dataset
# from finetuning.config import (
#     CKPT_DIR,
#     TOK_FILE,
#     TRAIN_SPLIT,
#     LEARNING_RATE,
#     BATCH_SIZE,
#     NUM_EPOCHS,
#     GRADIENT_ACCUMULATION_STEPS,
#     CHECKPOINT_DIR,
#     WEIGHT_DTYPE,
# )

# # ------------------------------------------------------------------
# # Custom TrainState
# # ------------------------------------------------------------------
# class TrainState(train_state.TrainState):
#     accum_grads: any
#     apply_fn: callable = field(pytree_node=False)
#     tx: optax.GradientTransformation = field(pytree_node=False)

# # ------------------------------------------------------------------
# # Loss
# # ------------------------------------------------------------------
# @jax.jit
# def compute_loss(logits, labels):
#     vocab = logits.shape[-1]
#     flat_logits = logits.reshape(-1, vocab)
#     flat_labels = labels.reshape(-1)
#     mask = flat_labels != -100
#     losses = optax.softmax_cross_entropy_with_integer_labels(
#         logits=flat_logits.astype(jnp.float32), labels=flat_labels
#     )
#     return jnp.sum(losses * mask) / (jnp.sum(mask) + 1e-8)

# # ------------------------------------------------------------------
# # Train step  (now wrapped with shard_map)
# # ------------------------------------------------------------------
# def _train_step_core(state, batch, rng_key, step):
#     rng = jax.random.fold_in(rng_key, step)

#     def loss_fn(p):
#         logits = state.apply_fn(
#             {"params": p},
#             tokens=batch["input_ids"],
#             segment_pos=batch["segment_pos"],
#             rngs={"dropout": rng},
#         )[0]
#         return compute_loss(logits, batch["labels"])

#     loss, grads = jax.value_and_grad(loss_fn)(state.params)
#     state = state.replace(
#         accum_grads=jtu.tree_map(lambda a, b: a + b, state.accum_grads, grads)
#     )
#     return state, loss

# def apply_grads(state):
#     grads = jtu.tree_map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
#     state = state.apply_gradients(grads=grads)
#     return state.replace(accum_grads=jtu.tree_map(jnp.zeros_like, state.accum_grads))

# # ------------------------------------------------------------------
# # Sharding helpers
# # ------------------------------------------------------------------
# def safe_fsdp_sharding(pytree, axis_name, num_devices):
#     """Shard only tensors whose axis-0 length divisible by num_devices."""
#     def spec(_, x):
#         if x.ndim >= 1 and x.shape[0] >= num_devices and x.shape[0] % num_devices == 0:
#             return PartitionSpec(axis_name, *([None] * (x.ndim - 1)))
#         return PartitionSpec()
#     return jtu.tree_map_with_path(spec, pytree)

# def print_tensor_stats(pytree, header):
#     if jax.process_index() != 0:
#         return
#     total = 0
#     print(header)
#     for path, x in jtu.tree_leaves_with_path(pytree):
#         total += x.size
#         print(f"  {path} -> {x.shape}  size={x.size}")
#     print(f"Total params: {total}")

# # ------------------------------------------------------------------
# # Main training loop
# # ------------------------------------------------------------------
# def main():
#     jax.distributed.initialize()
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Processes: {jax.process_count()}  Devices: {jax.device_count()}")
#         print("Devices:", jax.devices())

#     num_devices = jax.device_count()

#     # Simple FSDP mesh across all 8 cores
#     with Mesh(jax.devices(), axis_names=("fsdp",)) as mesh:

#         # 1) Load model
#         model, _, params, _ = load_recurrent_gemma_model(
#             CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE, use_checkpointing=False
#         )
#         if jax.process_index() == 0:
#             print_tensor_stats(params, "Parameter shapes")

#         # !!! ADD THIS DEBUG LINE !!!
#         if jax.process_index() == 0:
#             print(f"-----> SANITY CHECK: Model object configured with vocab_size = {model.config.vocab_size} <-----")


#         # 2) Dataset
#         ds, n_examples = get_dataset(TRAIN_SPLIT, BATCH_SIZE * jax.process_count())
#         ds = tfds.as_numpy(ds)
#         steps_per_epoch = (n_examples // (BATCH_SIZE * jax.process_count())) // GRADIENT_ACCUMULATION_STEPS
#         total_steps = steps_per_epoch * NUM_EPOCHS
#         if jax.process_index() == 0:
#             print(f"Total optimizer steps: {total_steps}")

#         # 3) Optimizer
#         lr = optax.cosine_decay_schedule(
#             init_value=LEARNING_RATE, decay_steps=total_steps, alpha=0.1
#         )
#         opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))

#         # 4) TrainState
#         state_cpu = TrainState.create(
#             apply_fn=model.apply,
#             params=params,
#             tx=opt,
#             accum_grads=jtu.tree_map(jnp.zeros_like, params),
#         )
#         del params

#         # 5) Safe FSDP sharding
#         sharding_spec = state_cpu.replace(
#             step=PartitionSpec(),
#             params=safe_fsdp_sharding(state_cpu.params, "fsdp", num_devices),
#             opt_state=safe_fsdp_sharding(state_cpu.opt_state, "fsdp", num_devices),
#             accum_grads=safe_fsdp_sharding(state_cpu.accum_grads, "fsdp", num_devices),
#         )
#         state = jax.device_put(
#             state_cpu,
#             jtu.tree_map(lambda s: NamedSharding(mesh, s), sharding_spec),
#         )

#         # 6) Data sharding
#         data_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))

#         # 7) Shard-map the two functions that invoke the Mosaic kernel
#         # p_train_step = shard_map(
#         #     _train_step_core,
#         #     mesh=mesh,
#         #     in_specs=(sharding_spec, data_sharding, None, None),
#         #     out_specs=(sharding_spec, None),
#         #     check_rep=False,
#         # )
#         # p_apply_grads = shard_map(
#         #     apply_grads,
#         #     mesh=mesh,
#         #     in_specs=(sharding_spec,),
#         #     out_specs=sharding_spec,
#         # )


#         # 7) Strip the NamedSharding wrappers → keep only the PartitionSpec
#         p_train_step = shard_map(
#             _train_step_core,
#             mesh=mesh,
#             in_specs=(sharding_spec, data_sharding.spec, None, None),  # .spec
#             out_specs=(sharding_spec, None),                           # .spec
#             check_rep=False,
#         )
#         p_apply_grads = shard_map(
#             apply_grads,
#             mesh=mesh,
#             in_specs=(sharding_spec,),         # sharding_spec is already a tree of specs
#             out_specs=sharding_spec,           # same here
# )

#         # 8) Training loop
#         rng = jax.random.PRNGKey(0)
#         rng = jax.random.fold_in(rng, jax.process_index())
#         ckpt_mgr = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())

#         for epoch in range(NUM_EPOCHS):
#             if jax.process_index() == 0:
#                 pbar = tqdm(ds, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
#             else:
#                 pbar = ds
#             tot = 0
#             for step, batch in enumerate(pbar):
#                 batch = jtu.tree_map(
#                     lambda x: multihost_utils.host_local_array_to_global_array(
#                         x, mesh, PartitionSpec("fsdp", *([None] * (x.ndim - 1)))
#                     ),
#                     batch,
#                 )
#                 state, loss = p_train_step(state, batch, rng, state.step)
#                 tot += loss

#                 if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                     state = p_apply_grads(state)
#                     if jax.process_index() == 0:
#                         avg = tot.item() / GRADIENT_ACCUMULATION_STEPS
#                         pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr(state.step):.6f}")
#                         tot = 0

#             # final flush
#             state = p_apply_grads(state)
#             jax.block_until_ready(state)

#         # 9) Save final checkpoint
#         if jax.process_index() == 0:
#             ckpt_mgr.save(step=state.step, items=state)
#             ckpt_mgr.wait_until_finished()
#             print("Training complete.")

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
Fool-proof training script for Recurrent-Gemma-2B on TPU v4-16
Full-parameter sharding (FSDP) with safe fall-back
"""

# 0) Force TPU and suppress CUDA noise
import os
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# 1) Core imports
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from flax.struct import field
from tqdm import tqdm
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.tree_util as jtu
import tensorflow_datasets as tfds
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map

# Corrected imports for recurrentgemma
import sentencepiece as spm
from recurrentgemma import jax as rg

from finetuning.data_pipeline import get_dataset
from finetuning.config import (
    CKPT_DIR,
    TOK_FILE,
    TRAIN_SPLIT,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    CHECKPOINT_DIR,
    WEIGHT_DTYPE,
)

# ------------------------------------------------------------------
# Custom TrainState
# ------------------------------------------------------------------
class TrainState(train_state.TrainState):
    accum_grads: any
    apply_fn: callable = field(pytree_node=False)
    tx: optax.GradientTransformation = field(pytree_node=False)

# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
@jax.jit
def compute_loss(logits, labels):
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    mask = flat_labels != -100
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=flat_logits.astype(jnp.float32), labels=flat_labels
    )
    return jnp.sum(losses * mask) / (jnp.sum(mask) + 1e-8)

# ------------------------------------------------------------------
# Train step
# ------------------------------------------------------------------
def _train_step_core(state, batch, rng_key, step):
    rng = jax.random.fold_in(rng_key, step)

    def loss_fn(p):
        logits = state.apply_fn(
            {"params": p},
            tokens=batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": rng},
        )[0]
        return compute_loss(logits, batch["labels"])

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.replace(
        accum_grads=jtu.tree_map(lambda a, b: a + b, state.accum_grads, grads)
    )
    return state, loss

def apply_grads(state):
    grads = jtu.tree_map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
    state = state.apply_gradients(grads=grads)
    return state.replace(accum_grads=jtu.tree_map(jnp.zeros_like, state.accum_grads))

# ------------------------------------------------------------------
# Sharding helpers
# ------------------------------------------------------------------
# def safe_fsdp_sharding(pytree, axis_name, num_devices):
#     """Shard only tensors whose axis-0 length divisible by num_devices."""
#     def spec(_, x):
#         if x.ndim >= 1 and x.shape[0] >= num_devices and x.shape[0] % num_devices == 0:
#             return PartitionSpec(axis_name, *([None] * (x.ndim - 1)))
#         return PartitionSpec()
#     return jtu.tree_map_with_path(spec, pytree)


# In train.py

def safe_fsdp_sharding(pytree, axis_name, num_devices):
    """
    Shards parameters using FSDP, but explicitly replicates the entire
    'embedder' module to avoid shape confusion bugs.
    """
    def spec(path, x):
        # This check correctly handles different key types
        is_in_embedder_module = any(
            p.key == 'embedder'
            for p in path
            if isinstance(p, jax.tree_util.DictKey)
        )

        if is_in_embedder_module:
            if jax.process_index() == 0:
                # This robustly prints the path for any PyTree structure
                path_str = '/'.join([
                    p.key if isinstance(p, jax.tree_util.DictKey) else f'[{p.idx}]'
                    for p in path
                ])
                print(f"--> Replicating parameter: {path_str}")
            return PartitionSpec() # Replicate

        # Apply standard FSDP sharding to all other parameters
        if x.ndim >= 1 and x.shape[0] >= num_devices and x.shape[0] % num_devices == 0:
            return PartitionSpec(axis_name, *([None] * (x.ndim - 1)))

        return PartitionSpec()

    return jtu.tree_map_with_path(spec, pytree)

def print_tensor_stats(pytree, header):
    if jax.process_index() != 0:
        return
    total = 0
    print(header)
    for path, x in jtu.tree_leaves_with_path(pytree):
        total += x.size
        print(f"  {path} -> {x.shape}  size={x.size}")
    print(f"Total params: {total}")

# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------
def main():
    jax.distributed.initialize()
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Processes: {jax.process_count()}  Devices: {jax.device_count()}")
        print("Devices:", jax.devices())

    num_devices = jax.device_count()

    with Mesh(jax.devices(), axis_names=("fsdp",)) as mesh:

        # --- CORRECTED MODEL LOADING PATTERN ---
        
        # 1. Load parameters from checkpoint FIRST
        print(f"Loading parameters from: {CKPT_DIR}")
        restored = ocp.PyTreeCheckpointer().restore(str(CKPT_DIR))
        params = restored.get("params", restored)
        if WEIGHT_DTYPE is not None:
            params = jax.tree.map(lambda x: x.astype(WEIGHT_DTYPE), params)

        if jax.process_index() == 0:
            print_tensor_stats(params, "Parameter shapes")
            
        # 2. Build the model config and instance SECOND
        print(f"Loading tokenizer from: {TOK_FILE}")
        vocab = spm.SentencePieceProcessor(model_file=str(TOK_FILE))
        
        # Use the corrected import path for Preset
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        
        # Use the corrected import path for GriffinConfig
        base_cfg = rg.GriffinConfig.from_preset(preset)
        cfg = base_cfg._replace(vocab_size=vocab.vocab_size())
        
        # The Griffin class from rg (recurrentgemma.jax) handles checkpointing
        model = rg.Griffin(cfg)
        
        if jax.process_index() == 0:
            print(f"-----> SANITY CHECK: Model object configured with vocab_size = {model.config.vocab_size} <-----")

        # 3. Dataset
        ds, n_examples = get_dataset(TRAIN_SPLIT, BATCH_SIZE * jax.process_count())
        ds = tfds.as_numpy(ds)
        steps_per_epoch = (n_examples // (BATCH_SIZE * jax.process_count())) // GRADIENT_ACCUMULATION_STEPS
        total_steps = steps_per_epoch * NUM_EPOCHS
        if jax.process_index() == 0:
            print(f"Total optimizer steps: {total_steps}")

        # 4. Optimizer
        lr = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE, decay_steps=total_steps, alpha=0.1
        )
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))

        # 5. Create TrainState THIRD, with params and model now correctly defined
        state_cpu = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=opt,
            accum_grads=jtu.tree_map(jnp.zeros_like, params),
        )
        del params

        # 6. Shard the state for FSDP
        sharding_spec = state_cpu.replace(
            step=PartitionSpec(),
            params=safe_fsdp_sharding(state_cpu.params, "fsdp", num_devices),
            opt_state=safe_fsdp_sharding(state_cpu.opt_state, "fsdp", num_devices),
            accum_grads=safe_fsdp_sharding(state_cpu.accum_grads, "fsdp", num_devices),
        )
        state = jax.device_put(
            state_cpu,
            jtu.tree_map(lambda s: NamedSharding(mesh, s), sharding_spec),
        )

        # 7. Prepare sharded functions
        data_sharding_spec = PartitionSpec("fsdp", None)
        p_train_step = shard_map(
            _train_step_core,
            mesh=mesh,
            in_specs=(sharding_spec, data_sharding_spec, None, None),
            out_specs=(sharding_spec, None),
            check_rep=False,
        )
        p_apply_grads = shard_map(
            apply_grads,
            mesh=mesh,
            in_specs=(sharding_spec,),
            out_specs=sharding_spec,
        )

        # 8. Training loop
        rng = jax.random.PRNGKey(0)
        rng = jax.random.fold_in(rng, jax.process_index())
        ckpt_mgr = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())

        for epoch in range(NUM_EPOCHS):
            if jax.process_index() == 0:
                pbar = tqdm(ds, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
            else:
                pbar = ds
            tot_loss = 0
            for step, batch in enumerate(pbar):
                batch = jtu.tree_map(
                    lambda x: multihost_utils.host_local_array_to_global_array(
                        x, mesh, PartitionSpec("fsdp", *([None] * (x.ndim - 1)))
                    ),
                    batch,
                )
                state, loss = p_train_step(state, batch, rng, state.step)
                tot_loss += loss

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    state = p_apply_grads(state)
                    if jax.process_index() == 0:
                        avg_loss = tot_loss.item() / GRADIENT_ACCUMULATION_STEPS
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr(state.step):.6f}")
                        tot_loss = 0

            # final flush at end of epoch
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
                state = p_apply_grads(state)
            jax.block_until_ready(state)

        # 9. Save final checkpoint
        if jax.process_index() == 0:
            ckpt_mgr.save(step=state.step, items=state)
            ckpt_mgr.wait_until_finished()
            print("Training complete.")

if __name__ == "__main__":
    main()
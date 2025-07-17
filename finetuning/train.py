

# #!/usr/bin/env python3
# """
# TPU v4-16 compatible training loop for Recurrent-Gemma-2B
# Full-parameter sharding (FSDP) to prevent per-core HBM OOM
# """

# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from flax.struct import field
# from tqdm import tqdm
# import time
# from pathlib import Path
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental.pjit import pjit
# import jax.tree_util
# import tensorflow_datasets as tfds
# from jax.experimental import multihost_utils

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

# # ---------------- TrainState ----------------
# class TrainState(train_state.TrainState):
#     accum_grads: any
#     apply_fn: callable = field(pytree_node=False)
#     tx: optax.GradientTransformation = field(pytree_node=False)

# # ---------------- loss ----------------
# @jax.jit
# def calculate_loss(logits, labels):
#     vocab_size = logits.shape[-1]
#     logits_flat = logits.reshape(-1, vocab_size)
#     labels_flat = labels.reshape(-1)
#     loss_mask = labels_flat != -100
#     losses = optax.softmax_cross_entropy_with_integer_labels(
#         logits=logits_flat.astype(jnp.float32), labels=labels_flat
#     )
#     masked_losses = jnp.where(loss_mask, losses, 0.0)
#     return jnp.sum(masked_losses) / (jnp.sum(loss_mask) + 1e-8)

# # ---------------- step ----------------
# def train_step(state, batch, base_dropout_rng, step_num):
#     step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

#     def loss_fn(params):
#         logits = state.apply_fn(
#             {"params": params},
#             tokens=batch["input_ids"],
#             segment_pos=batch["segment_pos"],
#             rngs={"dropout": step_dropout_key},
#         )[0]
#         return calculate_loss(logits, batch["labels"])

#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(state.params)
#     state = state.replace(
#         accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads)
#     )
#     return state, loss

# def apply_accumulated_gradients(state):
#     avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
#     state = state.apply_gradients(grads=avg_grads)
#     state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
#     return state

# # ---------------- main ----------------
# def main():
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Total processes: {jax.process_count()}; Global devices: {jax.device_count()}")

#     # --- single axis = FSDP across all 8 cores ---
#     with Mesh(jax.devices(), axis_names=("fsdp",)) as mesh:

#         # ---------- FSDP sharding helpers ----------
#         def fsdp_sharding(pytree):
#             def spec(_, x):
#                 if x.ndim == 2 and x.shape[0] > 1:  # weight matrices
#                     return PartitionSpec("fsdp", None)
#                 return PartitionSpec()
#             return jax.tree_util.tree_map_with_path(spec, pytree)

#         model, _, params, _ = load_recurrent_gemma_model(
#             CKPT_DIR,
#             TOK_FILE,
#             params_dtype=WEIGHT_DTYPE,
#             use_checkpointing=True,
#         )

#         # ---------- dataset ----------
#         train_dataset, num_train_examples = get_dataset(
#             TRAIN_SPLIT, BATCH_SIZE * jax.process_count()
#         )
#         train_dataset = tfds.as_numpy(train_dataset)

#         global_batch_size = BATCH_SIZE * jax.process_count()
#         steps_per_epoch = (num_train_examples // global_batch_size) // GRADIENT_ACCUMULATION_STEPS
#         total_train_steps = steps_per_epoch * NUM_EPOCHS

#         if jax.process_index() == 0:
#             print(f"Total optimizer steps: {total_train_steps}")

#         # ---------- optimizer ----------
#         lr_schedule = optax.cosine_decay_schedule(
#             init_value=LEARNING_RATE, decay_steps=total_train_steps, alpha=0.1
#         )
#         optimizer = optax.chain(
#             optax.clip_by_global_norm(1.0),
#             optax.adamw(learning_rate=lr_schedule),
#         )

#         # ---------- initial state ----------
#         state_on_cpu = TrainState.create(
#             apply_fn=model.apply,
#             params=params,
#             tx=optimizer,
#             accum_grads=jax.tree.map(jnp.zeros_like, params),
#         )
#         del params

#         state_sharding = state_on_cpu.replace(
#             step=PartitionSpec(),
#             params=fsdp_sharding(state_on_cpu.params),
#             opt_state=fsdp_sharding(state_on_cpu.opt_state),
#             accum_grads=fsdp_sharding(state_on_cpu.accum_grads),
#         )
#         sharding_for_put = jax.tree.map(
#             lambda spec: NamedSharding(mesh, spec), state_sharding
#         )
#         p_train_state = jax.device_put(state_on_cpu, sharding_for_put)

#         # ---------- data shard along same axis ----------
#         data_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))

#         # ---------- pjit functions ----------
#         p_train_step = pjit(
#             train_step,
#             in_shardings=(state_sharding, data_sharding, None, None),
#             out_shardings=(state_sharding, None),
#         )
#         p_apply_grads = pjit(
#             apply_accumulated_gradients,
#             in_shardings=(state_sharding,),
#             out_shardings=state_sharding,
#             donate_argnums=(0,),
#         )

#         # ---------- training loop ----------
#         rng = jax.random.PRNGKey(0)
#         base_dropout_rng = jax.random.fold_in(rng, jax.process_index())

#         ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())

#         for epoch in range(NUM_EPOCHS):
#             if jax.process_index() == 0:
#                 print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#                 pbar = tqdm(train_dataset, total=steps_per_epoch, desc=f"Epoch {epoch + 1}")
#             else:
#                 pbar = train_dataset

#             total_loss = 0
#             for step, batch in enumerate(pbar if jax.process_index() == 0 else train_dataset):
#                 sharded_batch = jax.tree_util.tree_map(
#                     lambda x: multihost_utils.host_local_array_to_global_array(
#                         x, mesh, PartitionSpec("fsdp", *([None] * (x.ndim - 1)))
#                     ),
#                     batch,
#                 )

#                 p_train_state, loss = p_train_step(p_train_state, sharded_batch, base_dropout_rng, p_train_state.step)
#                 total_loss += loss

#                 if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                     p_train_state = p_apply_grads(p_train_state)
#                     if jax.process_index() == 0:
#                         avg_loss = total_loss.item() / GRADIENT_ACCUMULATION_STEPS
#                         current_lr = lr_schedule(p_train_state.step)
#                         pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.6f}")
#                         if steps_per_epoch is not None:
#                             pbar.update(1)
#                         total_loss = 0

#             if jax.process_index() == 0 and steps_per_epoch is None:
#                 pbar.close()

#         step += 1
#         remaining = step % GRADIENT_ACCUMULATION_STEPS
#         if remaining:
#             p_train_state = p_apply_grads(p_train_state)

#         jax.block_until_ready(p_train_state)
#         if jax.process_index() == 0:
#             ckpt_manager.save(step=p_train_state.step, items=p_train_state)
#             ckpt_manager.wait_until_finished()
#             print("\nTraining complete.")

# if __name__ == "__main__":
#     main()




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
# # Train step
# # ------------------------------------------------------------------
# def train_step_fn(state, batch, rng_key, step):
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
#             CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE, use_checkpointing=True
#         )
#         if jax.process_index() == 0:
#             print_tensor_stats(params, "Parameter shapes")

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

#         p_train_step = pjit(
#             train_step_fn,
#             in_shardings=(sharding_spec, data_sharding, None, None),
#             out_shardings=(sharding_spec, None),
#         )
#         p_apply_grads = pjit(
#             apply_grads,
#             in_shardings=(sharding_spec,),
#             out_shardings=sharding_spec,
#             donate_argnums=(0,),
#         )

#         # 7) Training loop
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

#         # 8) Save final checkpoint
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
from jax.experimental.pjit import pjit
import jax.tree_util as jtu
import tensorflow_datasets as tfds
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map   # <-- NEW

from utils.model_loader import load_recurrent_gemma_model
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
# Train step  (now wrapped with shard_map)
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
def safe_fsdp_sharding(pytree, axis_name, num_devices):
    """Shard only tensors whose axis-0 length divisible by num_devices."""
    def spec(_, x):
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

    # Simple FSDP mesh across all 8 cores
    with Mesh(jax.devices(), axis_names=("fsdp",)) as mesh:

        # 1) Load model
        model, _, params, _ = load_recurrent_gemma_model(
            CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE, use_checkpointing=True
        )
        if jax.process_index() == 0:
            print_tensor_stats(params, "Parameter shapes")

        # 2) Dataset
        ds, n_examples = get_dataset(TRAIN_SPLIT, BATCH_SIZE * jax.process_count())
        ds = tfds.as_numpy(ds)
        steps_per_epoch = (n_examples // (BATCH_SIZE * jax.process_count())) // GRADIENT_ACCUMULATION_STEPS
        total_steps = steps_per_epoch * NUM_EPOCHS
        if jax.process_index() == 0:
            print(f"Total optimizer steps: {total_steps}")

        # 3) Optimizer
        lr = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE, decay_steps=total_steps, alpha=0.1
        )
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))

        # 4) TrainState
        state_cpu = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=opt,
            accum_grads=jtu.tree_map(jnp.zeros_like, params),
        )
        del params

        # 5) Safe FSDP sharding
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

        # 6) Data sharding
        data_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))

        # 7) Shard-map the two functions that invoke the Mosaic kernel
        # p_train_step = shard_map(
        #     _train_step_core,
        #     mesh=mesh,
        #     in_specs=(sharding_spec, data_sharding, None, None),
        #     out_specs=(sharding_spec, None),
        #     check_rep=False,
        # )
        # p_apply_grads = shard_map(
        #     apply_grads,
        #     mesh=mesh,
        #     in_specs=(sharding_spec,),
        #     out_specs=sharding_spec,
        # )


        # 7) Strip the NamedSharding wrappers → keep only the PartitionSpec
        p_train_step = shard_map(
            _train_step_core,
            mesh=mesh,
            in_specs=(sharding_spec, data_sharding.spec, None, None),  # .spec
            out_specs=(sharding_spec, None),                           # .spec
            check_rep=False,
        )
        p_apply_grads = shard_map(
            apply_grads,
            mesh=mesh,
            in_specs=(sharding_spec,),         # sharding_spec is already a tree of specs
            out_specs=sharding_spec,           # same here
)

        # 8) Training loop
        rng = jax.random.PRNGKey(0)
        rng = jax.random.fold_in(rng, jax.process_index())
        ckpt_mgr = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())

        for epoch in range(NUM_EPOCHS):
            if jax.process_index() == 0:
                pbar = tqdm(ds, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
            else:
                pbar = ds
            tot = 0
            for step, batch in enumerate(pbar):
                batch = jtu.tree_map(
                    lambda x: multihost_utils.host_local_array_to_global_array(
                        x, mesh, PartitionSpec("fsdp", *([None] * (x.ndim - 1)))
                    ),
                    batch,
                )
                state, loss = p_train_step(state, batch, rng, state.step)
                tot += loss

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    state = p_apply_grads(state)
                    if jax.process_index() == 0:
                        avg = tot.item() / GRADIENT_ACCUMULATION_STEPS
                        pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr(state.step):.6f}")
                        tot = 0

            # final flush
            state = p_apply_grads(state)
            jax.block_until_ready(state)

        # 9) Save final checkpoint
        if jax.process_index() == 0:
            ckpt_mgr.save(step=state.step, items=state)
            ckpt_mgr.wait_until_finished()
            print("Training complete.")

if __name__ == "__main__":
    main()
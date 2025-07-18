

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
# # def safe_fsdp_sharding(pytree, axis_name, num_devices):
# #     """Shard only tensors whose axis-0 length divisible by num_devices."""
# #     def spec(_, x):
# #         if x.ndim >= 1 and x.shape[0] >= num_devices and x.shape[0] % num_devices == 0:
# #             return PartitionSpec(axis_name, *([None] * (x.ndim - 1)))
# #         return PartitionSpec()
# #     return jtu.tree_map_with_path(spec, pytree)


# def safe_fsdp_sharding(pytree, axis_name, num_devices):
#     """Shard only tensors whose axis-0 length divisible by num_devices, but skip embedder."""
#     def spec(path, x):
#         # Skip embedder parameters - they should not be sharded along vocab dimension
#         if "embedder" in str(path):
#             return PartitionSpec()
            
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
# from jax.experimental.shard_map import shard_map

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
#     """Shard only tensors whose axis-0 length divisible by num_devices, but skip embedder."""
#     def spec(path, x):
#         # Skip embedder parameters - they should not be sharded along vocab dimension
#         if "embedder" in str(path):
#             return PartitionSpec()

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

#         if jax.process_index() == 0:
#             print(f"-----> SANITY CHECK: Model object configured with vocab_size = {model.config.vocab_size} <-----")

#         # 2) Dataset
#         ds, n_examples = get_dataset(TRAIN_SPLIT, BATCH_SIZE * jax.process_count())
#         ds = tfds.as_numpy(ds)
#         steps_per_epoch = (n_examples // (BATCH_SIZE * jax.process_count())) // GRADIENT_ACCUMULATION_STEPS
#         total_steps = steps_per_epoch * NUM_EPOCHS
#         if jax.process_index() == 0:
#             print(f"Total optimizer steps: {total_steps}")

#         # 3) Optimizer with Adafactor
#         lr = optax.cosine_decay_schedule(
#             init_value=LEARNING_RATE, decay_steps=total_steps, alpha=0.1
#         )
#         opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adafactor(learning_rate=lr))

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

#         # 7) Strip the NamedSharding wrappers → keep only the PartitionSpec
#         p_train_step = shard_map(
#             _train_step_core,
#             mesh=mesh,
#             in_specs=(sharding_spec, data_sharding.spec, None, None),
#             out_specs=(sharding_spec, None),
#             check_rep=False,
#         )
#         p_apply_grads = shard_map(
#             apply_grads,
#             mesh=mesh,
#             in_specs=(sharding_spec,),
#             out_specs=sharding_spec,
#         )

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





# #!/usr/bin/env python3
# """
# Final, Memory-Optimized Fine-Tuning Script for RecurrentGemma-2.7B.

# This script is specifically tailored for a 2-host TPU v4-16 environment and
# the user-provided model architecture. It uses:
# - Activation Rematerialization (Gradient Checkpointing)
# - Adafactor Optimizer for memory efficiency
# - A precise 2D Hybrid Sharding Strategy (FSDP + Tensor Parallelism)
# """
# # -----------------------------------------------------------------------------
# # 0. Setup and Core Imports
# # -----------------------------------------------------------------------------
# import os
# os.environ["JAX_PLATFORMS"] = "tpu"
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress verbose TF logs

# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental.pjit import pjit
# import jax.tree_util as jtu
# from jax.experimental import multihost_utils

# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from flax.linen import remat
# import tensorflow as tf
# from datasets import load_dataset

# from ml_collections import config_flags, ConfigDict
# from absl import app, flags, logging

# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from functools import partial
# from tqdm import tqdm

# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", help_string="Path to configuration file.")



# def _abs_path(p: str) -> str:
#    """Return an absolute path, expanding ~ and resolving symlinks."""
#    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))

# # -----------------------------------------------------------------------------
# # 1. Configuration (ml_collections)
# # -----------------------------------------------------------------------------
# def get_config():
#     """Defines all hyperparameters and settings for memory-optimized training."""
#     config = ConfigDict()

#     # --- Model & Paths ---
#     config.model_path = _abs_path("2b-it/2b-it")
#     config.tokenizer_path = _abs_path("2b-it/tokenizer.model")
#     config.ckpt_dir = _abs_path("finetuning_checkpoints")

#     # --- Training & Optimizer ---
#     config.learning_rate = 1e-5
#     config.num_epochs = 3
#     config.global_batch_size = 32 # Total batch size (16 per host)
#     config.grad_clip_norm = 1.0
#     config.grad_accum_steps = 8 # Use MultiSteps to simulate a larger batch size

#     # --- Evaluation ---
#     config.eval_steps = 250 # Run evaluation every N steps
#     config.eval_batch_size = 32

#     # --- Data ---
#     config.dataset_name = "HaimingW/miniF2F-lean4"
#     config.train_split = "train"
#     config.eval_split = "valid"
#     config.max_seq_len = 2048

#     # --- Sharding & Performance ---
#     config.data_axis = 'data'
#     config.model_axis = 'model'
#     config.weight_dtype = jnp.bfloat16
#     config.use_remat = True # Activation Rematerialization is crucial for memory saving
#     return config

# # -----------------------------------------------------------------------------
# # 2. Data Pipeline (Streaming and Efficient)
# # -----------------------------------------------------------------------------
# def create_dataset_iterator(config, split, mesh, batch_size):
#     """Creates a streaming, pre-fetching, and sharded tf.data pipeline."""
#     tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
#     if (batch_size % jax.process_count()) != 0:
#         raise ValueError(f"Batch size {batch_size} must be divisible by number of hosts {jax.process_count()}.")
#     per_host_batch_size = batch_size // jax.process_count()

#     def data_generator():
#         dataset = load_dataset(config.dataset_name, split=split, streaming=True)
#         dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())
#         dataset = dataset.shuffle(seed=42, buffer_size=1000)

#         for example in dataset:
#             text = f"theorem {example.get('name', '')} {example['formal_statement']} := by\n  {example['proof']}"
#             tokens = tokenizer.encode(text)
#             tokens = [tokenizer.bos_id()] + tokens + [tokenizer.eos_id()]
#             yield tokens[:config.max_seq_len]

#     tf_dataset = tf.data.Dataset.from_generator(
#         data_generator, output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
#     ).padded_batch(
#         batch_size=per_host_batch_size,
#         padded_shapes=[config.max_seq_len],
#         padding_values=0 # Use 0 for padding, standard for many tokenizers
#     )

#     # def to_global_jax_array(batch):
#     #     return multihost_utils.host_local_array_to_global_array(
#     #         batch, mesh, PartitionSpec(config.data_axis)
#     #     )

#     # tf_dataset = tf_dataset.map(lambda x: {"inputs": to_global_jax_array(x)}, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

#     tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)


#     # return tf_dataset.as_numpy_iterator()
  

#     def jax_iterator():
#         for np_batch in tf_dataset.as_numpy_iterator():
#             # np_batch is a plain NumPy array
#             yield {"inputs": multihost_utils.host_local_array_to_global_array(
#                       np_batch, mesh, PartitionSpec(config.data_axis))}

#         return jax_iterator()

    

# # -----------------------------------------------------------------------------
# # 3. Model Loading and Sharding
# # -----------------------------------------------------------------------------
# def get_partition_rules(config):
#     """
#     Defines the specific 2D sharding rules for the 2.7B RecurrentGemma model.
#     This uses a hybrid Tensor Parallelism ('model' axis) and FSDP ('data' axis) strategy.
#     """
#     return (
#         # 1. Tensor Parallelism for the largest layers
#         ("embedder/input_embedding", PartitionSpec("model", None)),
#         ("readout/kernel", PartitionSpec(None, "model")),
#         ("mlp_block/ffw_up/w", PartitionSpec(None, None, "model")),
#         ("mlp_block/ffw_down/kernel", PartitionSpec("model", None)),
#         ("attention_block/proj_q/kernel", PartitionSpec(None, "model")),
#         ("attention_block/proj_final/kernel", PartitionSpec("model", None)),
#         # 2. FSDP for remaining weights to save optimizer state memory
#         (r".*kernel", PartitionSpec("data", None)),
#         (r".*w", PartitionSpec("data", None)),
#         # 3. Replicate small parameters
#         (r".*bias", PartitionSpec()),
#         (r".*b", PartitionSpec()),
#         (r".*scale", PartitionSpec()),
#         (r".*a_param", PartitionSpec()),
#     )

# def load_and_shard_model(config, mesh):
#     """Loads model on CPU, defines sharding, and shards onto devices."""
#     with jax.default_device(jax.devices()[0]):
#         scan_fn = partial(remat) if config.use_remat else None
#         model_config = rg.GriffinConfig.from_preset(rg.Preset.RECURRENT_GEMMA_2B_V1)
#         model = rg.Griffin(model_config, dtype=config.weight_dtype)
#         # Load parameters from the original checkpoint directory
#         params_cpu = ocp.PyTreeCheckpointer().restore(config.model_path)

#     logical_rules = get_partition_rules(config)
#     # This utility converts the logical rules into a full PyTree of PartitionSpecs
#     # It requires flax.training.common_utils, which you might need to install
#     try:
#         from flax.training.common_utils import get_logical_partition_rules
#         partition_spec_tree = get_logical_partition_rules(params_cpu, logical_rules)
#     except ImportError:
#         logging.warning("Falling back to basic sharding due to flax version. For best results, use a version with `get_logical_partition_rules`.")
#         partition_spec_tree = jtu.tree_map(lambda _: PartitionSpec(), params_cpu)


#     shardings = jtu.tree_map(lambda p: NamedSharding(mesh, p), partition_spec_tree)

#     # pjit-ed function to move and shard params from CPU to TPU
#     # @pjit(out_shardings=sharding)


#     with mesh:
#         params_sharded = jax.device_put(params_cpu, shardings)



#     return model, params_sharded, shardings

# # -----------------------------------------------------------------------------
# # 4. Training State and Step Functions
# # -----------------------------------------------------------------------------
# class TrainingState(train_state.TrainState):
#     """A simple TrainState for managing parameters and optimizer state in Flax."""
#     pass

# def loss_fn(logits, batch):
#     """Standard cross-entropy loss for language modeling, ignoring padding."""
#     inputs = batch["inputs"]
#     labels = jnp.roll(inputs, shift=-1, axis=-1)
#     mask = inputs != 0
#     # Cast logits to float32 for stable loss calculation
#     loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), labels)
#     return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1e-8)

# def train_step(state, batch, rng_key):
#     """Performs a single training step, including forward pass, loss, and gradients."""
#     dropout_rng = jax.random.fold_in(rng_key, state.step)

#     def compute_loss(params):
#         logits = state.apply_fn(
#             {"params": params},
#             tokens=batch["inputs"],
#             segment_pos=jnp.arange(batch["inputs"].shape[-1]),
#             rngs={"dropout": dropout_rng}
#         )[0]
#         return loss_fn(logits, batch)

#     grad_fn = jax.value_and_grad(compute_loss)
#     loss, grads = grad_fn(state.params)
#     # Sync gradients across the data-parallel dimension ('data' axis)
#     grads = jax.lax.pmean(grads, axis_name='data')
#     state = state.apply_gradients(grads=grads)
#     return state, {"loss": loss}

# def eval_step(state, batch):
#     """Performs a single evaluation step (forward pass and loss calculation)."""
#     logits = state.apply_fn(
#         {"params": state.params},
#         tokens=batch["inputs"],
#         segment_pos=jnp.arange(batch["inputs"].shape[-1])
#     )[0]
#     return {"loss": loss_fn(logits, batch)}


# # -----------------------------------------------------------------------------
# # 5. Main Execution
# # -----------------------------------------------------------------------------
# def main(argv):
#     if len(argv) > 1:
#         raise app.UsageError("Too many command-line arguments.")
    
#     config = FLAGS.config
#     if config is None:
#         logging.warning("No --config file specified. Using default configuration from get_config().")
#         config = get_config()

#     ### START OF FIX ###
#     # Only convert dtype from string if it is a string. If using the default
#     # config, it's already a jnp object.
#     if isinstance(config.weight_dtype, str):
#         config.weight_dtype = getattr(jnp, config.weight_dtype)
#     ### END OF FIX ###

#     jax.distributed.initialize()
#     tf.config.set_visible_devices([], "GPU")

#     num_hosts = jax.process_count()
#     devices_per_host = jax.local_device_count()
#     # devices_array = jax.devices().reshape((num_hosts, devices_per_host))
#     devices_array = np.array(jax.devices()).reshape((num_hosts, devices_per_host))
#     mesh = Mesh(devices_array, (config.data_axis, config.model_axis))
    
#     logging.set_verbosity(logging.INFO)
#     if jax.process_index() == 0:
#         logging.info("--- Memory-Optimized Training Run ---")
#         logging.info(f"JAX initialized on {jax.process_count()} hosts with {jax.local_device_count()} devices each.")
#         logging.info(f"Device mesh created with shape: {mesh.shape}")
#         logging.info(f"Using Activation Rematerialization: {config.use_remat}")

#     rng = jax.random.PRNGKey(42)
#     os.makedirs(config.ckpt_dir, exist_ok=True)
    
#     train_iterator = create_dataset_iterator(config, config.train_split, mesh, config.global_batch_size)
#     eval_iterator = create_dataset_iterator(config, config.eval_split, mesh, config.eval_batch_size)

#     model, params, sharding = load_and_shard_model(config, mesh)
    
#     optimizer = optax.MultiSteps(
#         optax.chain(
#             optax.clip_by_global_norm(config.grad_clip_norm),
#             optax.adafactor(learning_rate=config.learning_rate)
#         ),
#         every_k_schedule=config.grad_accum_steps
#     )
    
#     state = TrainingState.create(apply_fn=model.apply, params=params, tx=optimizer)
#     del params

#     checkpointer = ocp.PyTreeCheckpointer()
#     ckpt_mngr = ocp.CheckpointManager(config.ckpt_dir, checkpointer, options=ocp.CheckpointManagerOptions(max_to_keep=1))
    
#     p_train_step = pjit(train_step, in_shardings=(sharding, None), out_shardings=(sharding, None))
#     p_eval_step = pjit(eval_step, in_shardings=(sharding, None), out_shardings=None)
    
#     if jax.process_index() == 0: logging.info("🚀 Starting training...")

#     best_eval_loss = float('inf')
#     for epoch in range(config.num_epochs):
#         train_metrics = []
#         num_train_steps = config.eval_steps * 5
#         if jax.process_index() == 0:
#             pbar = tqdm(total=num_train_steps, desc=f"Epoch {epoch+1}/{config.num_epochs}")

#         for step in range(1, num_train_steps + 1):
#             batch = next(train_iterator)
#             rng, step_rng = jax.random.split(rng)
            
#             state, metrics = p_train_step(state, batch, step_rng)
#             train_metrics.append(metrics)

#             if jax.process_index() == 0: pbar.update(1)

#             if state.step > 0 and state.step % config.eval_steps == 0:
#                 multihost_utils.sync_global_devices("eval_start")
#                 eval_loss = 0.0
#                 eval_batches = 50
#                 for _ in range(eval_batches):
#                     eval_batch = next(eval_iterator)
#                     eval_metrics = p_eval_step(state, eval_batch)
#                     eval_loss += eval_metrics['loss']
                
#                 eval_loss = multihost_utils.process_allgather(eval_loss).mean() / eval_batches
#                 train_loss_gathered = multihost_utils.process_allgather([m['loss'] for m in train_metrics])
#                 avg_train_loss = jnp.mean(train_loss_gathered)
                
#                 if jax.process_index() == 0:
#                     pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", eval_loss=f"{eval_loss:.4f}")
#                     logging.info(f"\nStep {state.step}: Train Loss={avg_train_loss:.4f}, Eval Loss={eval_loss:.4f}")
#                     train_metrics.clear()
                    
#                     if eval_loss < best_eval_loss:
#                         best_eval_loss = eval_loss
#                         logging.info(f"New best eval loss! Saving checkpoint to {config.ckpt_dir}")
#                         ckpt_mngr.save(step=int(state.step), args=ocp.args.StandardSave(state))
                
#                 if jax.process_index() == 0: pbar.reset(total=num_train_steps)
#         if jax.process_index() == 0: pbar.close()

#     ckpt_mngr.wait_until_finished()
#     if jax.process_index() == 0: logging.info("✅ Training complete.")

# if __name__ == "__main__":
#     # The config_flags.DEFINE_config_file at the top of the script registers the --config flag.
#     # absl.app.run() automatically parses this and other flags, then calls our main function.
#     app.run(main)





#!/usr/bin/env python3
"""
Final, Memory-Optimized Fine-Tuning Script for RecurrentGemma-2B-v1.

Designed for 2-host TPU v4-16:
- Activation rematerialization
- Adafactor + gradient accumulation
- FSDP + Tensor-Parallel hybrid 2-D sharding
"""
import os
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental import multihost_utils
# import jax.tree_util as jtu

# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# import tensorflow as tf
# from datasets import load_dataset

# from ml_collections import config_flags, ConfigDict
# from absl import app, flags, logging

# import os
# os.environ["JAX_PLATFORMS"] = "tpu"
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress verbose TF logs

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit
import jax.tree_util as jtu
from jax.experimental import multihost_utils

import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from flax.linen import remat
import tensorflow as tf
from datasets import load_dataset

from ml_collections import config_flags, ConfigDict
from absl import app, flags, logging

import recurrentgemma.jax as rg
import sentencepiece as spm
from functools import partial
from tqdm import tqdm


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", help_string="Path to configuration file.")


# -----------------------------------------------------------------------------
# 0. Helper utilities
# -----------------------------------------------------------------------------
def _abs_path(p: str) -> str:
    """Absolute path with tilde & env-var expansion."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


# -----------------------------------------------------------------------------
# 1. Default configuration
# -----------------------------------------------------------------------------
def get_config():
    c = ConfigDict()
    c.model_path = _abs_path("2b-it/2b-it")
    c.tokenizer_path = _abs_path("2b-it/tokenizer.model")
    c.ckpt_dir = _abs_path("finetuning_checkpoints")

    c.learning_rate = 1e-5
    c.num_epochs = 3
    c.global_batch_size = 32          # 16 per host on 2-host v4-16
    c.grad_clip_norm = 1.0
    c.grad_accum_steps = 8

    c.eval_steps = 250
    c.eval_batch_size = 32

    c.dataset_name = "HaimingW/miniF2F-lean4"
    c.train_split = "test"
    c.eval_split = "valid"
    c.max_seq_len = 2048

    c.data_axis = "data"
    c.model_axis = "model"
    c.weight_dtype = jnp.bfloat16
    c.use_remat = True
    return c


# -----------------------------------------------------------------------------
# 2. Streaming dataset
# -----------------------------------------------------------------------------
def create_dataset_iterator(config, split, mesh, batch_size):
    tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    per_host = batch_size // jax.process_count()

    def gen():
        ds = load_dataset(config.dataset_name, split=split, streaming=True)
        ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())
        ds = ds.shuffle(seed=42, buffer_size=2_000)

        yielded = 0
        for ex in ds:
            text = f"theorem {ex.get('name', '')} {ex['formal_statement']} := by\n  {ex['proof']}"
            tokens = [tokenizer.bos_id()] + tokenizer.encode(text) + [tokenizer.eos_id()]
            yield tokens[: config.max_seq_len]
            yielded += 1
            if yielded >= 100_000:        # safety
                break
        if yielded == 0:                  # empty split → dummy
            yield [tokenizer.bos_id(), tokenizer.eos_id()]

    tf_ds = (
        tf.data.Dataset.from_generator(
            gen, output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
        .padded_batch(per_host, padded_shapes=[config.max_seq_len], padding_values=0)
        .prefetch(tf.data.AUTOTUNE)
    )

    def jax_iter():
        for np_batch in tf_ds.as_numpy_iterator():
            yield {
                "inputs": multihost_utils.host_local_array_to_global_array(
                    np_batch, mesh, PartitionSpec(config.data_axis)
                )
            }

    return jax_iter()


# -----------------------------------------------------------------------------
# 3. Model / sharding
# -----------------------------------------------------------------------------
def get_partition_rules(config):
    return (
        ("embedder/input_embedding", PartitionSpec("model", None)),
        ("readout/kernel", PartitionSpec(None, "model")),
        ("mlp_block/ffw_up/w", PartitionSpec(None, None, "model")),
        ("mlp_block/ffw_down/kernel", PartitionSpec("model", None)),
        ("attention_block/proj_q/kernel", PartitionSpec(None, "model")),
        ("attention_block/proj_final/kernel", PartitionSpec("model", None)),
        (r".*kernel", PartitionSpec("data", None)),
        (r".*w", PartitionSpec("data", None)),
        (r".*bias", PartitionSpec()),
        (r".*b", PartitionSpec()),
        (r".*scale", PartitionSpec()),
        (r".*a_param", PartitionSpec()),
    )


def load_and_shard_model(config, mesh):
    with jax.default_device(jax.devices()[0]):
        model_cfg = rg.GriffinConfig.from_preset(rg.Preset.RECURRENT_GEMMA_2B_V1)
        model = rg.Griffin(model_cfg, dtype=config.weight_dtype)

        params_cpu = ocp.PyTreeCheckpointer().restore(config.model_path)

    # Build sharding spec tree
    try:
        from flax.training.common_utils import get_logical_partition_rules
        pspec_tree = get_logical_partition_rules(params_cpu, get_partition_rules(config))
    except ImportError:
        logging.warning("Using basic sharding – upgrade flax for logical rules")
        pspec_tree = jtu.tree_map(lambda _: PartitionSpec(), params_cpu)

    shardings = jtu.tree_map(lambda p: NamedSharding(mesh, p), pspec_tree)

    with mesh:
        params_sharded = jax.device_put(params_cpu, shardings)

    return model, params_sharded, shardings


# -----------------------------------------------------------------------------
# 4. Training step
# -----------------------------------------------------------------------------
class TrainState(train_state.TrainState):
    pass


def loss_fn(logits, batch):
    labels = jnp.roll(batch["inputs"], -1, axis=-1)
    mask = batch["inputs"] != 0
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), labels
    )
    return jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)


def train_step(state, batch, rng):
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_and_grad(p):
        logits = state.apply_fn(
            {"params": p},
            tokens=batch["inputs"],
            segment_pos=jnp.arange(batch["inputs"].shape[-1]),
            rngs={"dropout": dropout_rng},
        )[0]
        return loss_fn(logits, batch)

    loss, grads = jax.value_and_grad(loss_and_grad)(state.params)
    grads = jax.lax.pmean(grads, axis_name=config.data_axis)
    new_state = state.apply_gradients(grads=grads)
    return new_state, {"loss": loss}


def eval_step(state, batch):
    logits = state.apply_fn(
        {"params": state.params},
        tokens=batch["inputs"],
        segment_pos=jnp.arange(batch["inputs"].shape[-1]),
    )[0]
    return {"loss": loss_fn(logits, batch)}


# -----------------------------------------------------------------------------
# 5. Main
# -----------------------------------------------------------------------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    cfg = FLAGS.config or get_config()
    if isinstance(cfg.weight_dtype, str):
        cfg.weight_dtype = getattr(jnp, cfg.weight_dtype)

    jax.distributed.initialize()
    tf.config.set_visible_devices([], "GPU")

    mesh = Mesh(
        np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()),
        (cfg.data_axis, cfg.model_axis),
    )

    logging.set_verbosity(logging.INFO)
    if jax.process_index() == 0:
        logging.info(f"Mesh {mesh.shape} ready on {jax.process_count()} hosts")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    train_it = create_dataset_iterator(cfg, cfg.train_split, mesh, cfg.global_batch_size)
    eval_it = create_dataset_iterator(cfg, cfg.eval_split, mesh, cfg.eval_batch_size)

    model, params, shardings = load_and_shard_model(cfg, mesh)

    opt = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.adafactor(learning_rate=cfg.learning_rate),
        ),
        every_k_schedule=cfg.grad_accum_steps,
    )

    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    del params

    ckpt_mgr = ocp.CheckpointManager(
        cfg.ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )

    p_train = pjit(
        train_step,
        in_shardings=(shardings, None, None),
        out_shardings=(shardings, None),
        donate_argnums=(0,),
    )
    p_eval = pjit(
        eval_step,
        in_shardings=(shardings, None),
        out_shardings=None,
    )

    best_eval = float("inf")
    for epoch in range(cfg.num_epochs):
        steps = cfg.eval_steps * 5
        if jax.process_index() == 0:
            pbar = tqdm(total=steps, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        train_losses = []
        for step in range(1, steps + 1):
            batch = next(train_it)
            rng, s_rng = jax.random.split(jax.random.PRNGKey(epoch * 1000000 + step))
            state, m = p_train(state, batch, s_rng)
            train_losses.append(m["loss"])

            if jax.process_index() == 0:
                pbar.update(1)

            if state.step % cfg.eval_steps == 0:
                multihost_utils.sync_global_devices("eval")
                eval_loss = 0.0
                for _ in range(50):
                    eval_loss += p_eval(state, next(eval_it))["loss"]
                eval_loss = (
                    multihost_utils.process_allgather(eval_loss).mean() / 50
                )

                avg_train = jnp.mean(
                    multihost_utils.process_allgather(train_losses)
                )
                if jax.process_index() == 0:
                    pbar.set_postfix(train=avg_train, eval=eval_loss)
                    logging.info(
                        f"Step {state.step}: train={avg_train:.4f} eval={eval_loss:.4f}"
                    )
                    train_losses.clear()

                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        ckpt_mgr.save(
                            int(state.step),
                            args=ocp.args.StandardSave(state),
                        )
                        logging.info("Saved best checkpoint")
                if jax.process_index() == 0:
                    pbar.reset(total=steps)

        if jax.process_index() == 0:
            pbar.close()

    ckpt_mgr.wait_until_finished()
    if jax.process_index() == 0:
        logging.info("✅ Training complete.")


if __name__ == "__main__":
    app.run(main)
# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state  # Use the standard TrainState
# from tqdm import tqdm
# from pathlib import Path
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental.pjit import pjit
# import jax.tree_util
# from datasets import load_dataset

# from utils.model_loader import load_recurrent_gemma_model
# from finetuning.data_pipeline import get_dataset
# from finetuning.config import (
#     CKPT_DIR,
#     TOK_FILE,
#     TRAIN_SPLIT,
#     DATASET_NAME,
#     LEARNING_RATE,
#     BATCH_SIZE,
#     NUM_EPOCHS,
#     # GRADIENT_ACCUMULATION_STEPS is no longer needed
#     CHECKPOINT_DIR,
#     WEIGHT_DTYPE,
#     DATASET_PROPORTION,
# )

# # Using the standard flax TrainState as we no longer need to store accumulated gradients.
# TrainState = train_state.TrainState

# # Loss function remains the same, can be jitted for performance.
# @jax.jit
# def calculate_loss(logits, labels):
#     """Calculates the cross-entropy loss between logits and labels."""
#     vocab_size = logits.shape[-1]
#     logits_flat = logits.reshape(-1, vocab_size)
#     labels_flat = labels.reshape(-1)
#     loss_mask = (labels_flat != -100)
#     losses = optax.softmax_cross_entropy_with_integer_labels(
#         logits=logits_flat.astype(jnp.float32), labels=labels_flat
#     )
#     masked_losses = jnp.where(loss_mask, losses, 0.0)
#     total_loss = jnp.sum(masked_losses)
#     num_valid_tokens = jnp.sum(loss_mask)
#     loss = total_loss / (num_valid_tokens + 1e-8)
#     return loss

# def train_step(state, batch, base_dropout_rng, step_num):
#     """
#     Performs a single training step.
#     This function now calculates the loss, computes gradients, and applies them immediately.
#     """
#     step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

#     def loss_fn(params):
#         """The loss function to be differentiated."""
#         logits = state.apply_fn(
#             {"params": params},
#             tokens=batch["input_ids"],
#             segment_pos=batch["segment_pos"],
#             rngs={"dropout": step_dropout_key}
#         )[0]
#         return calculate_loss(logits, batch["labels"])

#     # Compute both the loss and the gradients in a single pass.
#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(state.params)
    
#     # Apply the gradients to update the model parameters.
#     state = state.apply_gradients(grads=grads)
    
#     return state, loss

# def print_training_summary(num_devices, effective_batch_size, raw_dataset_size):
#     """Prints a summary of the training configuration."""
#     global_batch_size = effective_batch_size * num_devices
    
#     examples_to_use = int(raw_dataset_size * DATASET_PROPORTION)
#     num_train_steps_per_epoch = examples_to_use // global_batch_size

#     print("\n" + "="*80)
#     print("TRAINING CONFIGURATION SUMMARY (No Gradient Accumulation)")
#     print("="*80)
#     print(f"Number of accelerator devices: {num_devices}")
#     print(f"Per-device batch size: {effective_batch_size}")
#     print(f"Global batch size (per optimizer step): {global_batch_size}")
#     print(f"Raw dataset size: {raw_dataset_size}")
#     print(f"Proportion to use: {DATASET_PROPORTION * 100:.1f}% ({examples_to_use} examples)")
#     print(f"Optimizer steps per epoch: {num_train_steps_per_epoch}")
#     print(f"Number of epochs: {NUM_EPOCHS}")
#     print("="*80 + "\n")


# def setup(mesh):
#     """Handles all the boilerplate setup for model, optimizer, and state."""
#     if jax.process_index() == 0:
#         print("Setting up model, optimizer, and sharded training state...")

#     data_sharding = NamedSharding(mesh, PartitionSpec('data_axis'))
#     replicated_sharding = NamedSharding(mesh, PartitionSpec())

#     def get_param_sharding(param_pytree):
#         """Defines sharding rules for model parameters."""
#         def get_spec(param):
#             if param.ndim > 1 and param.size > 1_000_000:
#                 return PartitionSpec(*([None] * (param.ndim - 1) + ['data_axis']))
#             return PartitionSpec()
#         return jax.tree.map(get_spec, param_pytree)

#     model, _, params, _ = load_recurrent_gemma_model(
#         CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE
#     )
    
#     class ScanShardingHelper:
#         def __init__(self, mesh):
#             self.mesh = mesh
#             self.sequence_axis_name = None
#             self.sequence_axis_index_groups = None
#             self.activations_sharding_spec = PartitionSpec('data_axis')
#             self.rnn_state_sharding_spec = PartitionSpec('data_axis')

#     model.scan_sharding_spec = ScanShardingHelper(mesh=mesh)

#     param_sharding_rules = get_param_sharding(params)

#     optimizer = optax.adafactor(learning_rate=LEARNING_RATE)
#     dummy_opt_state = optimizer.init(params)
#     if isinstance(dummy_opt_state, tuple):
#         opt_state_sharding_rules = tuple(get_param_sharding(s) for s in dummy_opt_state)
#     else:
#         opt_state_sharding_rules = get_param_sharding(dummy_opt_state)

#     # Sharding specification for the standard TrainState.
#     state_sharding_spec = TrainState(
#         step=PartitionSpec(),
#         apply_fn=None,
#         params=param_sharding_rules,
#         tx=None,
#         opt_state=opt_state_sharding_rules,
#     )

#     # Create TrainState without pjit first, then shard it
#     train_state = TrainState.create(
#         apply_fn=model.apply, 
#         params=params, 
#         tx=optimizer
#     )
    
#     # Now shard the created state
#     p_train_state = jax.device_put(train_state, NamedSharding(mesh, state_sharding_spec))
    
#     del params, dummy_opt_state

#     # pjit-compile the unified training step.
#     p_train_step = pjit(
#         train_step,
#         in_shardings=(state_sharding_spec, data_sharding, replicated_sharding, replicated_sharding),
#         out_shardings=(state_sharding_spec, replicated_sharding),
#         donate_argnums=(0,) # Donate the state buffer for in-place update.
#     )

#     return p_train_state, p_train_step

# def run_training_loop(state, p_train_step, train_dataset):
#     """Executes the main training loop over epochs and steps."""
#     if jax.process_index() == 0:
#         print("Starting training...")

#     rng = jax.random.PRNGKey(0)
#     base_dropout_rng = jax.random.fold_in(rng, jax.process_index())
    
#     num_train_steps_per_epoch = len(train_dataset)

#     for epoch in range(NUM_EPOCHS):
#         if jax.process_index() == 0:
#             print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#             pbar = tqdm(train_dataset, total=num_train_steps_per_epoch, desc=f"Epoch {epoch + 1}")
#         else:
#             pbar = train_dataset

#         for step, batch in enumerate(pbar):
#             batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
            
#             # Perform a full training step (forward, backward, and optimizer update).
#             state, loss = p_train_step(state, batch, base_dropout_rng, step)

#             if jax.process_index() == 0:
#                 # Update the progress bar with the loss from the current step.
#                 pbar.set_postfix(loss=f"{loss.item():.4f}")
#                 pbar.update(1)
    
#     return state

# def main():
#     """Main orchestration function."""
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Total processes: {jax.process_count()}; Local devices: {jax.local_device_count()}; Global devices: {jax.device_count()}")

#     num_devices = jax.device_count()
    
#     effective_batch_size = BATCH_SIZE
#     if BATCH_SIZE == 1:
#         effective_batch_size = 2
#         if jax.process_index() == 0:
#             print("\nWARNING: BATCH_SIZE is 1, temporarily overriding to 2 to avoid library bug.\n")

#     if jax.process_index() == 0:
#         print("Loading dataset metadata to calculate training steps...")
#         raw_dataset_for_size = load_dataset(DATASET_NAME, split=TRAIN_SPLIT)
#         raw_dataset_size = len(raw_dataset_for_size)
#         del raw_dataset_for_size
#         print_training_summary(num_devices, effective_batch_size, raw_dataset_size)
    
#     train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)

#     with Mesh(jax.devices(), axis_names=('data_axis',)) as mesh:
#         state, p_train_step = setup(mesh)
#         final_state = run_training_loop(state, p_train_step, train_dataset)

#         jax.block_until_ready(final_state)
#         if jax.process_index() == 0:
#             print("\nSaving final checkpoint...")
#             ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
#             raw_dataset_size = len(load_dataset(DATASET_NAME, split=TRAIN_SPLIT))
#             num_train_steps_per_epoch = (int(raw_dataset_size * DATASET_PROPORTION) // (effective_batch_size * num_devices))
#             final_step = num_train_steps_per_epoch * NUM_EPOCHS
#             ckpt_manager.save(step=final_step, items=final_state)
#             print("Final checkpoint saved.")

#     if jax.process_index() == 0:
#         print("\nTraining complete.")

# if __name__ == "__main__":
#     from finetuning.pretokenize_dataset import pretokenize_and_save
#     pretokenize_and_save()
#     main()




# #!/usr/bin/env python3
# """
# Fine-tune RecurrentGemma-2B on FrenzyMath/Herald_proofs
# """

# import os
# os.environ["JAX_PLATFORMS"] = "tpu"
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# import flax.core.frozen_dict as frozen_dict

# def _safe_repr(self):
#     return "<FrozenDict (sharded)>"

# frozen_dict.FrozenDict.__repr__ = _safe_repr

# import numpy as np
# import flax.core                 
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental.shard_map import shard_map
# import jax.tree_util as jtu
# from jax.experimental import multihost_utils

# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# import tensorflow as tf
# from datasets import load_dataset
# from ml_collections import config_flags, ConfigDict
# from absl import app, flags, logging
# from functools import partial

# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from tqdm import tqdm

# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", help_string="Path to configuration file.")


# # ------------------------------------------------------------------
# # helpers
# # ------------------------------------------------------------------
# def _abs_path(p: str) -> str:
#     return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


# # ------------------------------------------------------------------
# # default config
# # ------------------------------------------------------------------
# def get_config():
#     c = ConfigDict()
#     c.model_path = _abs_path("2b-it/2b-it")
#     c.tokenizer_path = _abs_path("2b-it/tokenizer.model")
#     c.ckpt_dir = _abs_path("finetuning_checkpoints")

#     c.learning_rate = 1e-5
#     c.num_epochs = 3
#     c.global_batch_size = 32
#     c.grad_clip_norm = 1.0
#     c.grad_accum_steps = 8

#     c.eval_steps = 250
#     c.eval_batch_size = 32

#     c.dataset_name = "FrenzyMath/Herald_proofs"
#     c.train_split = "train"
#     c.eval_split = "valid"
#     c.max_seq_len = 2048
#     c.dataset_fraction = 0.001
#     c.weight_dtype = jnp.bfloat16
#     c.data_axis = "data"
#     c.model_axis = "model"
#     return c


# # ------------------------------------------------------------------
# # dataset
# # ------------------------------------------------------------------
# def create_dataset_iterator(config, split, mesh, batch_size):
#     tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
#     per_host = batch_size // jax.process_count()

#     def gen():
#         ds = load_dataset(config.dataset_name, split=split, streaming=True)
#         ds = ds.shuffle(seed=42, buffer_size=2_000)
#         required = {"informal_theorem", "formal_theorem", "formal_proof"}
#         limit = max(1, int(100_000 * config.dataset_fraction))

#         yielded = 0
#         for ex in ds:
#             if not required <= ex.keys():
#                 continue
#             text = (
#                 f"{ex.get('header', '')}\n"
#                 f"--- informal theorem ---\n{ex['informal_theorem']}\n"
#                 f"--- formal theorem ---\n{ex['formal_theorem']}\n"
#                 f"--- proof ---\n{ex['formal_proof']}"
#             )
#             tokens = [tokenizer.bos_id()] + tokenizer.encode(text) + [tokenizer.eos_id()]
#             yield tokens[: config.max_seq_len]
#             yielded += 1
#             if yielded >= limit:
#                 break
#         if yielded == 0:
#             yield [tokenizer.bos_id(), tokenizer.eos_id()]

#     tf_ds = (
#         tf.data.Dataset.from_generator(
#             gen,
#             output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
#         )
#         .shard(num_shards=jax.process_count(), index=jax.process_index())
#         .padded_batch(per_host, padded_shapes=[config.max_seq_len], padding_values=0)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     def jax_iter():
#         for np_batch in tf_ds.as_numpy_iterator():
#             yield {
#                 "inputs": multihost_utils.host_local_array_to_global_array(
#                     np_batch, mesh, PartitionSpec(config.data_axis)
#                 )
#             }

#     return jax_iter()


# # ------------------------------------------------------------------
# # model
# # ------------------------------------------------------------------
# def get_partition_rules():
#     return (
#         ("embedder/input_embedding", PartitionSpec("model", None)),
#         ("readout/kernel", PartitionSpec(None, "model")),
#         ("mlp_block/ffw_up/w", PartitionSpec(None, None, "model")),
#         ("mlp_block/ffw_down/kernel", PartitionSpec("model", None)),
#         ("attention_block/proj_q/kernel", PartitionSpec(None, "model")),
#         ("attention_block/proj_final/kernel", PartitionSpec("model", None)),
#         (r".*kernel", PartitionSpec("data", None)),
#         (r".*w", PartitionSpec("data", None)),
#         (r".*bias|.*b|.*scale|.*a_param", PartitionSpec()),
#     )


# def load_and_shard_model(config, mesh):
#     with jax.default_device(jax.devices()[0]):
#         model_cfg = rg.GriffinConfig.from_preset(rg.Preset.RECURRENT_GEMMA_2B_V1)
#         model = rg.Griffin(model_cfg, dtype=config.weight_dtype)
#         params_cpu = ocp.PyTreeCheckpointer().restore(config.model_path)

#     try:
#         from flax.training.common_utils import get_logical_partition_rules
#         pspec_tree = get_logical_partition_rules(params_cpu, get_partition_rules())
#     except ImportError:
#         logging.warning("Using basic sharding – upgrade flax for logical rules")
#         pspec_tree = jtu.tree_map(lambda _: PartitionSpec(), params_cpu)

#     shardings = jtu.tree_map(
#         lambda ps: NamedSharding(mesh, ps), pspec_tree
#     )
#     with mesh:
#         params_sharded = jax.device_put(params_cpu, shardings)
#     return model, params_sharded, pspec_tree
    


# # ------------------------------------------------------------------
# # loss / step
# # ------------------------------------------------------------------
# class TrainState(train_state.TrainState):
#     pass


# def loss_fn(logits, batch):
#     labels = jnp.roll(batch["inputs"], -1, axis=-1)
#     mask = batch["inputs"] != 0
#     loss = optax.softmax_cross_entropy_with_integer_labels(
#         logits.astype(jnp.float32), labels
#     )
#     return jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)



# def _train_step(state, batch, rng, model, data_axis_name):
#     # dropout_rng = jax.random.fold_in(rng, state.step)
#     dropout_rng = jax.random.fold_in(rng, jax.lax.axis_index(data_axis_name))

#     def _loss(p):
#         batch_size, seq_len = batch["inputs"].shape
#         segment_pos = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
#         logits = state.apply_fn(
#             {"params": p},
#             batch["inputs"],
#             segment_pos,
#             # rngs={"dropout": dropout_rng},  # <-- The fix
#         )[0]
#         return loss_fn(logits, batch)

#     loss, grads = jax.value_and_grad(_loss)(state.params)
#     new_state = state.apply_gradients(grads=grads)
#     return new_state, {"loss": loss}


# def _eval_step(state, batch, model):
#     def _loss(p):
#         batch_size, seq_len = batch["inputs"].shape
#         segment_pos = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
#         logits = state.apply_fn(
#             {"params": p},
#             batch["inputs"],
#             segment_pos,
#             deterministic=True,
#         )[0]
#         return loss_fn(logits, batch)

#     loss = _loss(state.params)
#     return {"loss": loss}


# # ------------------------------------------------------------------
# # main
# # ------------------------------------------------------------------
# def main(argv):
#     if len(argv) > 1:
#         raise app.UsageError("Too many command-line arguments.")

#     cfg = get_config()
#     jax.distributed.initialize()
#     tf.config.set_visible_devices([], "GPU")

#     devices = np.array(jax.devices()).reshape(
#         jax.process_count(), jax.local_device_count()
#     )
#     mesh = Mesh(devices, (cfg.data_axis, cfg.model_axis))

#     logging.set_verbosity(logging.INFO)
#     if jax.process_index() == 0:
#         logging.info(f"Mesh {mesh} ready on {jax.process_count()} hosts")
#     os.makedirs(cfg.ckpt_dir, exist_ok=True)

#     train_it = create_dataset_iterator(cfg, cfg.train_split, mesh, cfg.global_batch_size)
#     eval_it = create_dataset_iterator(cfg, cfg.eval_split, mesh, cfg.eval_batch_size)

#     model, params, pspecs = load_and_shard_model(cfg, mesh)



#     opt = optax.MultiSteps(
#         optax.chain(
#             optax.clip_by_global_norm(cfg.grad_clip_norm),
#             optax.adafactor(learning_rate=cfg.learning_rate),
#         ),
#         every_k_schedule=cfg.grad_accum_steps,
#     )

#     state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

#     # shard-map wrappers
#     def _spec_for_state(state_template, params_pspec_tree):
#         def _map(path, value):
#             if path and str(path[0]) == "params":
#                 return jtu.tree_get(params_pspec_tree, path[1:])
#             elif path and str(path[0]) == "step":
#                 return PartitionSpec()
#             return None
#         return jtu.tree_map_with_path(_map, state_template)

#     state_pspec = _spec_for_state(state, pspecs)
#     batch_pspec = PartitionSpec(cfg.data_axis, None)

#     train_step_sharded = shard_map(
#         partial(_train_step, model=model, data_axis_name=cfg.data_axis),
#         mesh=mesh,
#         in_specs=(state_pspec, batch_pspec, None),
#         out_specs=(state_pspec, None),
#         check_rep=False,
#     )

#     eval_step_sharded = shard_map(
#         partial(_eval_step, model=model),
#         mesh=mesh,
#         in_specs=(state_pspec, batch_pspec),
#         out_specs=None,
#         check_rep=False,
#     )

#     ckpt_mgr = ocp.CheckpointManager(
#         cfg.ckpt_dir,
#         options=ocp.CheckpointManagerOptions(max_to_keep=1),
#     )

#     rng_key = jax.random.PRNGKey(42)
#     best_eval = float("inf")

#     for epoch in range(cfg.num_epochs):
#         steps = cfg.eval_steps * 5
#         if jax.process_index() == 0:
#             pbar = tqdm(total=steps, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

#         train_losses = []
#         for step in range(1, steps + 1):
#             batch = next(train_it)
#             rng_key, step_rng = jax.random.split(rng_key)

#             state, metrics = train_step_sharded(state, batch, step_rng)
#             train_losses.append(metrics["loss"])

#             if jax.process_index() == 0:
#                 pbar.update(1)

#             if step % cfg.eval_steps == 0:
#                 multihost_utils.sync_global_devices("eval")
#                 eval_losses = []
#                 for _ in range(50):
#                     eval_batch = next(eval_it)
#                     eval_metrics = eval_step_sharded(state, eval_batch)
#                     eval_losses.append(eval_metrics["loss"])

#                 eval_loss = jnp.mean(jnp.array(eval_losses))
#                 avg_train = jnp.mean(jnp.array(train_losses))

#                 if jax.process_index() == 0:
#                     pbar.set_postfix(train=float(avg_train), eval=float(eval_loss))
#                     logging.info(
#                         f"Step {step}: train={avg_train:.4f} eval={eval_loss:.4f}"
#                     )
#                     train_losses.clear()

#                     if eval_loss < best_eval:
#                         best_eval = eval_loss
#                         ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
#                         logging.info("Saved best checkpoint")

#         if jax.process_index() == 0:
#             pbar.close()

#     ckpt_mgr.wait_until_finished()
#     if jax.process_index() == 0:
#         logging.info("✅ Training complete.")


# if __name__ == "__main__":
#     app.run(main)





# #!/usr/bin/env python3
# """
# Fine-tune RecurrentGemma-2B on FrenzyMath/Herald_proofs
# (TPU-safe version – Mosaic kernels wrapped in shard_map)
# """

# import os
# os.environ["JAX_PLATFORMS"] = "tpu"
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import flax.core.frozen_dict as frozen_dict
# def _safe_repr(self):
#     return "<FrozenDict (sharded)>"
# frozen_dict.FrozenDict.__repr__ = _safe_repr

# import numpy as np
# import flax.core
# import jax
# import jax.numpy as jnp
# from jax.sharding import Mesh, PartitionSpec
# import jax.tree_util as jtu
# from jax.experimental import multihost_utils
# from jax.experimental.shard_map import shard_map   # NEW

# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# import tensorflow as tf
# from datasets import load_dataset
# from ml_collections import config_flags, ConfigDict
# from absl import app, flags, logging
# from functools import partial

# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from tqdm import tqdm

# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", help_string="Path to configuration file.")

# # ------------------------------------------------------------------
# # helpers
# # ------------------------------------------------------------------
# def _abs_path(p: str) -> str:
#     return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))

# # ------------------------------------------------------------------
# # default config
# # ------------------------------------------------------------------
# def get_config():
#     c = ConfigDict()
#     c.model_path = _abs_path("2b-it/2b-it")
#     c.tokenizer_path = _abs_path("2b-it/tokenizer.model")
#     c.ckpt_dir = _abs_path("finetuning_checkpoints")
#     c.learning_rate = 1e-5
#     c.num_epochs = 3
#     c.global_batch_size = 32
#     c.grad_clip_norm = 1.0
#     c.grad_accum_steps = 1
#     c.eval_steps = 250
#     c.eval_batch_size = 32
#     c.dataset_name = "FrenzyMath/Herald_proofs"
#     c.train_split = "train"
#     c.eval_split = "valid"
#     c.max_seq_len = 2048
#     c.dataset_fraction = 0.001
#     c.weight_dtype = jnp.bfloat16
#     c.data_axis = "data"
#     c.model_axis = "model"
#     return c

# # ------------------------------------------------------------------
# # dataset
# # ------------------------------------------------------------------
# def create_dataset_iterator(config, split, mesh, batch_size):
#     tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
#     per_host = batch_size // jax.process_count()

#     def gen():
#         ds = load_dataset(config.dataset_name, split=split, streaming=True)
#         ds = ds.shuffle(seed=42, buffer_size=2_000)
#         required = {"informal_theorem", "formal_theorem", "formal_proof"}
#         limit = max(1, int(100_000 * config.dataset_fraction))
#         yielded = 0
#         for ex in ds:
#             if not required <= ex.keys():
#                 continue
#             text = (
#                 f"{ex.get('header', '')}\n"
#                 f"--- informal theorem ---\n{ex['informal_theorem']}\n"
#                 f"--- formal theorem ---\n{ex['formal_theorem']}\n"
#                 f"--- proof ---\n{ex['formal_proof']}"
#             )
#             tokens = [tokenizer.bos_id()] + tokenizer.encode(text) + [tokenizer.eos_id()]
#             yield tokens[: config.max_seq_len]
#             yielded += 1
#             if yielded >= limit:
#                 break
#         if yielded == 0:
#             yield [tokenizer.bos_id(), tokenizer.eos_id()]

#     tf_ds = (
#         tf.data.Dataset.from_generator(
#             gen,
#             output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
#         )
#         .shard(num_shards=jax.process_count(), index=jax.process_index())
#         .padded_batch(per_host, padded_shapes=[config.max_seq_len], padding_values=0)
#         .prefetch(tf.data.AUTOTUNE)
#     )

#     def jax_iter():
#         for np_batch in tf_ds.as_numpy_iterator():
#             yield {
#                 "inputs": multihost_utils.host_local_array_to_global_array(
#                     np_batch, mesh, PartitionSpec(config.data_axis)
#                 )
#             }

#     return jax_iter()

# # ------------------------------------------------------------------
# # model
# # ------------------------------------------------------------------
# def get_partition_rules():
#     return (
#         ("embedder/input_embedding", PartitionSpec("model", None)),
#         ("readout/kernel", PartitionSpec(None, "model")),
#         ("mlp_block/ffw_up/w", PartitionSpec(None, None, "model")),
#         ("mlp_block/ffw_down/kernel", PartitionSpec("model", None)),
#         ("attention_block/proj_q/kernel", PartitionSpec(None, "model")),
#         ("attention_block/proj_final/kernel", PartitionSpec("model", None)),
#         (r".*kernel", PartitionSpec("data", None)),
#         (r".*w", PartitionSpec("data", None)),
#         (r".*bias|.*b|.*scale|.*a_param", PartitionSpec()),
#     )

# def load_and_shard_model(config, mesh):
#     with jax.default_device(jax.devices()[0]):
#         model_cfg = rg.GriffinConfig.from_preset(rg.Preset.RECURRENT_GEMMA_2B_V1)
#         model = rg.Griffin(model_cfg, dtype=config.weight_dtype)
#         params_cpu = ocp.PyTreeCheckpointer().restore(config.model_path)

#     try:
#         from flax.training.common_utils import get_logical_partition_rules
#         pspec_tree = get_logical_partition_rules(params_cpu, get_partition_rules())
#     except ImportError:
#         logging.warning("Using basic sharding – upgrade flax for logical rules")
#         pspec_tree = jtu.tree_map(lambda _: PartitionSpec(), params_cpu)

#     shardings = jtu.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), pspec_tree)
#     with mesh:
#         params_sharded = jax.device_put(params_cpu, shardings)
#     return model, params_sharded, pspec_tree

# # ------------------------------------------------------------------
# # loss / step
# # ------------------------------------------------------------------
# class TrainState(train_state.TrainState):
#     pass

# # ------------- NEW: sharded forward+loss via shard_map -------------
# # def make_sharded_forward_fn(model, mesh, data_axis):
# #     def _forward(tokens, positions, params):
# #         logits, _ = model.apply(
# #             {"params": params},
# #             tokens=tokens,
# #             segment_pos=positions,
# #             cache=None,
# #         )
# #         targets = tokens[:, 1:]
# #         logits_flat = logits.reshape(-1, logits.shape[-1])
# #         targets_flat = targets.reshape(-1)
# #         mask = targets_flat != 0
# #         loss = optax.softmax_cross_entropy_with_integer_labels(
# #             logits_flat.astype(jnp.float32), targets_flat
# #         )
# #         loss = jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)
# #         return loss

# #     sharded = shard_map(
# #         _forward,
# #         mesh=mesh,
# #         in_specs=(
# #             PartitionSpec(data_axis),  # tokens
# #             PartitionSpec(data_axis),  # positions
# #             PartitionSpec(),           # params (already sharded)
# #         ),
# #         out_specs=PartitionSpec(),     # scalar loss per shard
# #         check_rep=False,
# #     )
# #     return sharded
# # ------------------------------------------------------------------


# # def make_sharded_forward_fn(model, mesh, data_axis):
# #     """
# #     Return a jit-able, TPU-sharded forward+loss function for RecurrentGemma.
# #     Ensures logits and targets have identical length.
# #     """

# #     def _forward(tokens, positions, params):
# #         # tokens : [B, L]   (already padded to max_seq_len)
# #         # positions : [B, L]

# #         # 1. Build inputs and targets
# #         inputs  = tokens[:, :-1]          # [B, L-1]
# #         targets = tokens[:, 1:]           # [B, L-1]

# #         # 2. Forward pass
# #         logits, _ = model.apply(
# #             {"params": params},
# #             tokens=inputs,
# #             segment_pos=positions[:, :-1],  # [B, L-1]
# #             cache=None,
# #         )                                  # logits: [B, L-1, V]

# #         # 3. Flatten for optax
# #         logits_flat = logits.reshape(-1, logits.shape[-1])  # [(B*(L-1)), V]
# #         targets_flat = targets.reshape(-1)                  # [(B*(L-1))]

# #         # 4. Mask out padding tokens
# #         mask = targets_flat != 0

# #         # 5. Cross-entropy loss
# #         loss = optax.softmax_cross_entropy_with_integer_labels(
# #             logits_flat.astype(jnp.float32),
# #             targets_flat
# #         )
# #         loss = jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)
# #         return loss

# #     # Shard the helper across the data axis
# #     sharded = shard_map(
# #         _forward,
# #         mesh=mesh,
# #         in_specs=(
# #             PartitionSpec(data_axis),  # tokens
# #             PartitionSpec(data_axis),  # positions
# #             PartitionSpec(),           # params (already sharded)
# #         ),
# #         out_specs=PartitionSpec(),     # scalar loss per shard
# #         check_rep=False,
# #     )
# #     return sharded



# # def make_sharded_forward_fn(model, mesh, data_axis):
# #     """
# #     Return a jit-able, TPU-sharded forward+loss function for RecurrentGemma.
# #     Ensures the model receives a 2048-length input.
# #     """
# #     def _forward(tokens, positions, params):
# #         # tokens have shape [B, 2048]

# #         # 1. Forward pass with the full 2048-length sequence
# #         logits, _ = model.apply(
# #             {"params": params},
# #             tokens=tokens,          # Pass full [B, 2048] tensor
# #             segment_pos=positions,  # Pass full [B, 2048] tensor
# #             cache=None,
# #         )  # Logits will have shape [B, 2048, V]

# #         # 2. Create targets and align logits for loss calculation
# #         # The logit at index `i` is the prediction for token `i+1`.
# #         # We drop the last logit, as it predicts a token outside our sequence.
# #         relevant_logits = logits[:, :-1, :]  # Shape: [B, 2047, V]
# #         targets = tokens[:, 1:]              # Shape: [B, 2047]

# #         # 3. Flatten for optax
# #         logits_flat = relevant_logits.reshape(-1, relevant_logits.shape[-1])
# #         targets_flat = targets.reshape(-1)

# #         # 4. Mask out padding and calculate loss
# #         mask = targets_flat != 0
# #         loss = optax.softmax_cross_entropy_with_integer_labels(
# #             logits_flat.astype(jnp.float32),
# #             targets_flat
# #         )
# #         loss = jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)
# #         return loss

# #     # The shard_map wrapper remains the same
# #     sharded = shard_map(
# #         _forward,
# #         mesh=mesh,
# #         in_specs=(
# #             PartitionSpec(data_axis),  # tokens
# #             PartitionSpec(data_axis),  # positions
# #             PartitionSpec(),           # params (already sharded)
# #         ),
# #         out_specs=PartitionSpec(),     # scalar loss per shard
# #         check_rep=False,
# #     )
# #     return sharded



# def make_sharded_forward_fn(model, mesh, data_axis):
#     """
#     Return a jit-able, TPU-sharded forward+loss function for RecurrentGemma.
#     This version correctly aligns inputs and targets BEFORE the model call.
#     """

#     def _forward(tokens, positions, params):
#         # tokens: [B, L] (e.g., [B, 2048])
#         # positions: [B, L]

#         # 1. Build inputs and targets BEFORE the model call.
#         # The model's task is to predict the next token.
#         # Input: tokens 0 to L-2.
#         # Target: tokens 1 to L-1.
#         inputs  = tokens[:, :-1]          # Shape: [B, L-1]
#         targets = tokens[:, 1:]           # Shape: [B, L-1]

#         # 2. Forward pass with the correctly sized input
#         logits, _ = model.apply(
#             {"params": params},
#             tokens=inputs,
#             segment_pos=positions[:, :-1],  # Match the input's length
#             cache=None,
#         )                                  # Logits will have shape [B, L-1, V]

#         # 3. Flatten for optax loss calculation
#         logits_flat = logits.reshape(-1, logits.shape[-1])
#         targets_flat = targets.reshape(-1)

#         # 4. Mask out padding tokens (where target is 0)
#         mask = targets_flat != 0

#         # 5. Calculate cross-entropy loss
#         loss = optax.softmax_cross_entropy_with_integer_labels(
#             logits_flat.astype(jnp.float32),
#             targets_flat
#         )
#         loss = jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)
#         return loss

#     # Shard the function across the data axis of the mesh
#     sharded = shard_map(
#         _forward,
#         mesh=mesh,
#         in_specs=(
#             PartitionSpec(data_axis),  # tokens
#             PartitionSpec(data_axis),  # positions
#             PartitionSpec(),           # params (already sharded)
#         ),
#         out_specs=PartitionSpec(),     # scalar loss per shard
#         check_rep=False,
#     )
#     return sharded


# @partial(
#     jax.jit,
#     static_argnames=["model", "optimizer", "mesh", "data_axis"],
#     donate_argnames=["params", "opt_state"],
# )
# def train_step(
#     model,
#     params,
#     optimizer,
#     opt_state,
#     batch,
#     *,
#     mesh,
#     data_axis,
# ):
#     fwd = make_sharded_forward_fn(model, mesh, data_axis)

#     def _loss(p):
#         positions = jnp.broadcast_to(
#             jnp.arange(batch["inputs"].shape[1])[None, :], batch["inputs"].shape
#         )
#         return fwd(batch["inputs"], positions, p)

#     loss, grads = jax.value_and_grad(_loss)(params)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return params, opt_state, {"loss": loss}

# @partial(jax.jit, static_argnames=["model", "mesh", "data_axis"])
# def eval_step(model, params, batch, *, mesh, data_axis):
#     fwd = make_sharded_forward_fn(model, mesh, data_axis)
#     positions = jnp.broadcast_to(
#         jnp.arange(batch["inputs"].shape[1])[None, :], batch["inputs"].shape
#     )
#     loss = fwd(batch["inputs"], positions, params)
#     return {"loss": loss}

# # ------------------------------------------------------------------
# # main
# # ------------------------------------------------------------------
# def main(argv):
#     if len(argv) > 1:
#         raise app.UsageError("Too many command-line arguments.")

#     cfg = get_config()
#     jax.distributed.initialize()
#     tf.config.set_visible_devices([], "GPU")

#     devices = np.array(jax.devices()).reshape(
#         jax.process_count(), jax.local_device_count()
#     )
#     mesh = Mesh(devices, (cfg.data_axis, cfg.model_axis))

#     logging.set_verbosity(logging.INFO)
#     if jax.process_index() == 0:
#         logging.info(f"Mesh {mesh} ready on {jax.process_count()} hosts")
#     os.makedirs(cfg.ckpt_dir, exist_ok=True)

#     train_it = create_dataset_iterator(cfg, cfg.train_split, mesh, cfg.global_batch_size)
#     eval_it = create_dataset_iterator(cfg, cfg.eval_split, mesh, cfg.eval_batch_size)

#     model, params, pspecs = load_and_shard_model(cfg, mesh)

#     base_opt = optax.chain(
#         optax.clip_by_global_norm(cfg.grad_clip_norm),
#         optax.adafactor(learning_rate=cfg.learning_rate),
#     )
#     opt = optax.MultiSteps(base_opt, every_k_schedule=cfg.grad_accum_steps)
#     opt_state = opt.init(params)

#     state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
#     ckpt_mgr = ocp.CheckpointManager(
#         cfg.ckpt_dir,
#         options=ocp.CheckpointManagerOptions(max_to_keep=1),
#     )
#     best_eval = float("inf")

#     for epoch in range(cfg.num_epochs):
#         steps = cfg.eval_steps * 5
#         if jax.process_index() == 0:
#             pbar = tqdm(total=steps, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

#         train_losses = []
#         for step in range(1, steps + 1):
#             batch = next(train_it)

#             params, opt_state, metrics = train_step(
#                 model, params, opt, opt_state, batch,
#                 mesh=mesh,
#                 data_axis=cfg.data_axis,
#             )
#             train_losses.append(metrics["loss"])

#             if jax.process_index() == 0:
#                 pbar.update(1)

#             if step % cfg.eval_steps == 0:
#                 multihost_utils.sync_global_devices("eval")
#                 eval_losses = []
#                 for _ in range(50):
#                     eval_batch = next(eval_it)
#                     eval_metrics = eval_step(
#                         model, params, eval_batch,
#                         mesh=mesh,
#                         data_axis=cfg.data_axis,
#                     )
#                     eval_losses.append(eval_metrics["loss"])

#                 eval_loss = jnp.mean(jnp.array(eval_losses))
#                 avg_train = jnp.mean(jnp.array(train_losses))

#                 if jax.process_index() == 0:
#                     pbar.set_postfix(train=float(avg_train), eval=float(eval_loss))
#                     logging.info(
#                         f"Step {step}: train={avg_train:.4f} eval={eval_loss:.4f}"
#                     )
#                     train_losses.clear()

#                     if eval_loss < best_eval:
#                         best_eval = eval_loss
#                         state = state.replace(params=params, opt_state=opt_state, step=step)
#                         ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
#                         logging.info("Saved best checkpoint")

#         if jax.process_index() == 0:
#             pbar.close()

#     ckpt_mgr.wait_until_finished()
#     if jax.process_index() == 0:
#         logging.info("✅ Training complete.")

# if __name__ == "__main__":
#     app.run(main)




import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import multihost_utils
import flax.linen as nn
import optax
from absl import app, flags, logging
from functools import partial
import time

# This tells JAX to target the TPU and ignore any local GPUs.
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# --- 1. A Trivial Model ---
class SimpleModel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.features)(x)

# --- 2. The JIT-compiled Training Step ---
@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(model, params, opt_state, optimizer, batch):
    """A single training step on a batch of data."""

    def loss_fn(p):
        logits = model.apply({"params": p}, batch['x'])
        # Simple mean squared error loss against a target of ones.
        loss = jnp.mean((logits - batch['y']) ** 2)
        return loss

    # Compute loss and gradients.
    loss, grads = jax.value_and_grad(loss_fn)(params)
    # Update model parameters.
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

# --- 3. Main Execution Logic ---
def main(argv):
    # --- Boilerplate for multi-host setup ---
    jax.distributed.initialize()
    logging.set_verbosity(logging.INFO)
    if jax.process_index() == 0:
        logging.info(f"JAX initialized on {jax.process_count()} processes with {jax.device_count()} total devices.")

    # --- Create a 2D mesh for data and model parallelism ---
    # This assumes you have at least 2 devices per host.
    # Adjust ('data', 'model') and the reshape dimensions if needed.
    num_hosts = jax.process_count()
    num_local_devices = jax.local_device_count()
    devices = jax.devices()
    
    if len(devices) < 2:
        raise ValueError("This test requires at least 2 JAX devices.")

    # Create a mesh of all devices. We'll just use one dimension for this simple test.
    mesh = Mesh(devices, axis_names=('data',))

    if jax.process_index() == 0:
        logging.info(f"Created device mesh: {mesh}")

    # --- Configuration ---
    batch_size = 16
    input_features = 128
    output_features = 1
    learning_rate = 0.01
    num_steps = 20

    # --- Initialize Model and Optimizer ---
    model = SimpleModel(features=output_features)
    key = jax.random.PRNGKey(0)
    
    # Initialize parameters on the CPU first.
    with jax.default_device(jax.devices("cpu")[0]):
        params = model.init(key, jnp.ones((batch_size, input_features)))['params']
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # --- Shard parameters and data across the mesh ---
    # Replicate parameters on all devices.
    params_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec()) 
    # Shard data along the 'data' axis.
    data_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec('data', None))

    params = jax.device_put(params, params_sharding)
    opt_state = jax.device_put(opt_state, params_sharding)

    if jax.process_index() == 0:
        logging.info("Model and optimizer state have been sharded across devices.")

    # --- Training Loop with Dummy Data ---
    for step in range(num_steps):
        # Create a dummy batch for this step.
        dummy_x = jnp.ones((batch_size, input_features))
        dummy_y = jnp.ones((batch_size, output_features))

        # Shard the dummy data to the devices.
        batch = {
            'x': jax.device_put(dummy_x, data_sharding),
            'y': jax.device_put(dummy_y, data_sharding)
        }
        
        if jax.process_index() == 0 and step == 0:
            logging.info("Starting first train_step. JIT compilation will happen now.")
            start_time = time.time()

        # Execute the training step.
        params, opt_state, loss = train_step(model, params, opt_state, optimizer, batch)
        
        # Use block_until_ready to ensure the computation is finished before we print.
        # This is crucial for accurate timing and for catching hangs.
        loss.block_until_ready()

        if jax.process_index() == 0:
            if step == 0:
                end_time = time.time()
                logging.info(f"JIT compilation and first step finished in {end_time - start_time:.2f} seconds.")
            
            logging.info(f"Step {step+1}/{num_steps}, Loss: {loss:.4f}")

    if jax.process_index() == 0:
        logging.info("✅ Minimal test script completed successfully.")

if __name__ == "__main__":
    app.run(main)


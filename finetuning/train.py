

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
# Fine-tune RecurrentGemma-2B on FrenzyMath/Herald_proofs
# (Lean 4) with a configurable subset.
# """


# import os
# os.environ["JAX_PLATFORMS"] = "tpu"
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
#     c.data_axis = 'data'
#     c.model_axis = 'model'
#     return c


# # ------------------------------------------------------------------
# # streaming dataset
# # ------------------------------------------------------------------
# def create_dataset_iterator(config, split, mesh, batch_size):
#     tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
#     per_host = batch_size // jax.process_count()

#     def gen():
#         ds = load_dataset(config.dataset_name, split=split, streaming=True)
#         ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())
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
#         if yielded == 0:  # empty shard → dummy
#             yield [tokenizer.bos_id(), tokenizer.eos_id()]

#     tf_ds = (
#         tf.data.Dataset.from_generator(
#             gen, output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
#         )
#         .padded_batch(
#             per_host, padded_shapes=[config.max_seq_len], padding_values=0
#         )
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
# # model / sharding
# # ------------------------------------------------------------------
# def get_partition_rules(config):
#     return (
#         ("embedder/input_embedding", PartitionSpec("model", None)),
#         ("readout/kernel", PartitionSpec(None, "model")),
#         ("mlp_block/ffw_up/w", PartitionSpec(None, None, "model")),
#         ("mlp_block/ffw_down/kernel", PartitionSpec("model", None)),
#         ("attention_block/proj_q/kernel", PartitionSpec(None, "model")),
#         ("attention_block/proj_final/kernel", PartitionSpec("model", None)),
#         (r".*kernel", PartitionSpec("data", None)),
#         (r".*w", PartitionSpec("data", None)),
#         (r".*bias", PartitionSpec()),
#         (r".*b", PartitionSpec()),
#         (r".*scale", PartitionSpec()),
#         (r".*a_param", PartitionSpec()),
#     )


# def load_and_shard_model(config, mesh):
#     with jax.default_device(jax.devices()[0]):
#         model_cfg = rg.GriffinConfig.from_preset(rg.Preset.RECURRENT_GEMMA_2B_V1)
#         model = rg.Griffin(model_cfg, dtype=config.weight_dtype)
#         params_cpu = ocp.PyTreeCheckpointer().restore(config.model_path)

#     try:
#         from flax.training.common_utils import get_logical_partition_rules
#         pspec_tree = get_logical_partition_rules(params_cpu, get_partition_rules(config))
#     except ImportError:
#         logging.warning("Using basic sharding – upgrade flax for logical rules")
#         pspec_tree = jtu.tree_map(lambda _: PartitionSpec(), params_cpu)

#     shardings = jtu.tree_map(lambda p: NamedSharding(mesh, p), pspec_tree)
#     with mesh:
#         params_sharded = jax.device_put(params_cpu, shardings)
#     return model, params_sharded, shardings


# # ------------------------------------------------------------------
# # training step
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


# def train_step(state, batch, rng):
#     dropout_rng = jax.random.fold_in(rng, state.step)

#     def loss_and_grad(p):
#         logits = state.apply_fn(
#             {"params": p},
#             tokens=batch["inputs"],
#             segment_pos=jnp.arange(batch["inputs"].shape[-1]),
#             rngs={"dropout": dropout_rng},
#         )[0]
#         return loss_fn(logits, batch)

#     loss, grads = jax.value_and_grad(loss_and_grad)(state.params)
#     grads = jax.lax.pmean(grads, axis_name=config.data_axis)
#     new_state = state.apply_gradients(grads=grads)
#     return new_state, {"loss": loss}


# def eval_step(state, batch):
#     logits = state.apply_fn(
#         {"params": state.params},
#         tokens=batch["inputs"],
#         segment_pos=jnp.arange(batch["inputs"].shape[-1]),
#     )[0]
#     return {"loss": loss_fn(logits, batch)}


# # ------------------------------------------------------------------
# # main
# # ------------------------------------------------------------------
# def main(argv):
#     if len(argv) > 1:
#         raise app.UsageError("Too many command-line arguments.")

#     cfg = get_config()

#     jax.distributed.initialize()
#     tf.config.set_visible_devices([], "GPU")

#     mesh = Mesh(
#         np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()),
#         (cfg.data_axis, cfg.model_axis),
#     )

#     logging.set_verbosity(logging.INFO)
#     if jax.process_index() == 0:
#         logging.info(f"Mesh {mesh.shape} ready on {jax.process_count()} hosts")

#     os.makedirs(cfg.ckpt_dir, exist_ok=True)

#     train_it = create_dataset_iterator(cfg, cfg.train_split, mesh, cfg.global_batch_size)
#     eval_it = create_dataset_iterator(cfg, cfg.eval_split, mesh, cfg.eval_batch_size)

#     model, params, shardings = load_and_shard_model(cfg, mesh)

#     opt = optax.MultiSteps(
#         optax.chain(
#             optax.clip_by_global_norm(cfg.grad_clip_norm),
#             optax.adafactor(learning_rate=cfg.learning_rate),
#         ),
#         every_k_schedule=cfg.grad_accum_steps,
#     )

#     state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
#     del params

#     ckpt_mgr = ocp.CheckpointManager(
#         cfg.ckpt_dir,
#         options=ocp.CheckpointManagerOptions(max_to_keep=1),
#     )

#     p_train = jax.jit(
#         train_step,
#         in_shardings=(shardings, None, None),
#         out_shardings=(shardings, None),
#         donate_argnums=(0,),
#     )
#     p_eval = jax.jit(
#         eval_step,
#         in_shardings=(shardings, None),
#         out_shardings=None,
#     )

#     best_eval = float("inf")
#     for epoch in range(cfg.num_epochs):
#         steps = cfg.eval_steps * 5
#         if jax.process_index() == 0:
#             pbar = tqdm(total=steps, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

#         train_losses = []
#         for step in range(1, steps + 1):
#             batch = next(train_it)
#             rng, s_rng = jax.random.split(jax.random.PRNGKey(epoch * 1_000_000 + step))
#             state, m = p_train(state, batch, s_rng)
#             train_losses.append(m["loss"])

#             if jax.process_index() == 0:
#                 pbar.update(1)

#             if state.step % cfg.eval_steps == 0:
#                 multihost_utils.sync_global_devices("eval")
#                 eval_loss = 0.0
#                 for _ in range(50):
#                     eval_loss += p_eval(state, next(eval_it))["loss"]
#                 eval_loss = (
#                     multihost_utils.process_allgather(eval_loss).mean() / 50
#                 )

#                 avg_train = jnp.mean(
#                     multihost_utils.process_allgather(train_losses)
#                 )
#                 if jax.process_index() == 0:
#                     pbar.set_postfix(train=avg_train, eval=eval_loss)
#                     logging.info(
#                         f"Step {state.step}: train={avg_train:.4f} eval={eval_loss:.4f}"
#                     )
#                     train_losses.clear()

#                     if eval_loss < best_eval:
#                         best_eval = eval_loss
#                         ckpt_mgr.save(
#                             int(state.step),
#                             args=ocp.args.StandardSave(state),
#                         )
#                         logging.info("Saved best checkpoint")
#                 if jax.process_index() == 0:
#                     pbar.reset(total=steps)

#         if jax.process_index() == 0:
#             pbar.close()

#     ckpt_mgr.wait_until_finished()
#     if jax.process_index() == 0:
#         logging.info("✅ Training complete.")


# if __name__ == "__main__":
#     app.run(main)




#!/usr/bin/env python3
"""
Fine-tune RecurrentGemma-2B on FrenzyMath/Herald_proofs
(Lean 4) with a configurable subset.
"""

import os
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from flax.core import freeze
from jax.tree_util import tree_map_with_path

from ml_collections import config_flags, ConfigDict
from absl import app, flags, logging

import recurrentgemma.jax as rg
import sentencepiece as spm
from functools import partial
from tqdm import tqdm

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", help_string="Path to configuration file.")

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))


def _make_state_sharding(state, params_shardings):
    """Return a TrainState whose leaves are either:
       - the corresponding sharding for params, or
       - None (=> replicated) for every other field.
    """
    def _map(path, value):
        # If this leaf sits under state.params, use the supplied sharding.
        if path and path[0] == "params":
            return jtu.tree_get(params_shardings, path[1:])
        return None  # replicate everything else
    return jtu.tree_map_with_path(_map, state)

# ------------------------------------------------------------------
# default config
# ------------------------------------------------------------------
def get_config():
    c = ConfigDict()
    c.model_path = _abs_path("2b-it/2b-it")
    c.tokenizer_path = _abs_path("2b-it/tokenizer.model")
    c.ckpt_dir = _abs_path("finetuning_checkpoints")

    c.learning_rate = 1e-5
    c.num_epochs = 3
    c.global_batch_size = 32
    c.grad_clip_norm = 1.0
    c.grad_accum_steps = 8

    c.eval_steps = 250
    c.eval_batch_size = 32

    c.dataset_name = "FrenzyMath/Herald_proofs"
    c.train_split = "train"
    c.eval_split = "valid"
    c.max_seq_len = 2048
    c.dataset_fraction = 0.001
    c.weight_dtype = jnp.bfloat16
    c.data_axis = 'data'
    c.model_axis = 'model'
    return c

# ------------------------------------------------------------------
# streaming dataset
# ------------------------------------------------------------------
def create_dataset_iterator(config, split, mesh, batch_size):
    tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    per_host = batch_size // jax.process_count()

    def gen():
        ds = load_dataset(config.dataset_name, split=split, streaming=True)
        # do NOT shard here – we will shard the *generator* later
        ds = ds.shuffle(seed=42, buffer_size=2_000)

        required = {"informal_theorem", "formal_theorem", "formal_proof"}
        limit = max(1, int(100_000 * config.dataset_fraction))

        yielded = 0
        for ex in ds:
            if not required <= ex.keys():
                continue
            text = (
                f"{ex.get('header', '')}\n"
                f"--- informal theorem ---\n{ex['informal_theorem']}\n"
                f"--- formal theorem ---\n{ex['formal_theorem']}\n"
                f"--- proof ---\n{ex['formal_proof']}"
            )
            tokens = [tokenizer.bos_id()] + tokenizer.encode(text) + [tokenizer.eos_id()]
            yield tokens[: config.max_seq_len]
            yielded += 1
            if yielded >= limit:
                break

        # If this rank produced nothing, emit one dummy sample
        if yielded == 0:
            yield [tokenizer.bos_id(), tokenizer.eos_id()]

    tf_ds = (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
        .shard(num_shards=jax.process_count(), index=jax.process_index())
        .padded_batch(
            per_host,
            padded_shapes=[config.max_seq_len],
            padding_values=0,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    def jax_iter():
        tf_iter = tf_ds.as_numpy_iterator()
        while True:
            try:
                np_batch = next(tf_iter)
            except StopIteration:
                # Dataset exhausted – repeat dummy batch
                np_batch = np.zeros((per_host, config.max_seq_len), dtype=np.int32)

            yield {
                "inputs": multihost_utils.host_local_array_to_global_array(
                    np_batch, mesh, PartitionSpec(config.data_axis)
                )
            }

    return jax_iter()

# ------------------------------------------------------------------
# model / sharding
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# training step
# ------------------------------------------------------------------
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
    grads = jax.lax.pmean(grads, axis_name=FLAGS.config.data_axis)
    new_state = state.apply_gradients(grads=grads)
    return new_state, {"loss": loss}


def eval_step(state, batch):
    logits = state.apply_fn(
        {"params": state.params},
        tokens=batch["inputs"],
        segment_pos=jnp.arange(batch["inputs"].shape[-1]),
    )[0]
    return {"loss": loss_fn(logits, batch)}


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    cfg = get_config()
    

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


    state_sharding = _make_state_sharding(state, shardings)

    p_train = jax.jit(
        train_step,
        in_shardings=(state_sharding, None, None),
        out_shardings=(state_sharding, None),
        donate_argnums=(0,),
    )

    p_eval = jax.jit(
        eval_step,
        in_shardings=(state_sharding, None),
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
            rng, s_rng = jax.random.split(jax.random.PRNGKey(epoch * 1_000_000 + step))
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
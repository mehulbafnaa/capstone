

#!/usr/bin/env python3
"""
Fine-tune RecurrentGemma-2B on FrenzyMath/Herald_proofs
(TPU-safe version – Mosaic kernels wrapped in shard_map)
"""

import os
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import flax.core.frozen_dict as frozen_dict
def _safe_repr(self):
    return "<FrozenDict (sharded)>"
frozen_dict.FrozenDict.__repr__ = _safe_repr

import numpy as np
import flax.core
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import jax.tree_util as jtu
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map

import optax
import orbax.checkpoint as ocp
from flax.training import train_state
import tensorflow as tf
from datasets import load_dataset
from ml_collections import config_flags, ConfigDict
from absl import app, flags, logging
from functools import partial

import recurrentgemma.jax as rg
import sentencepiece as spm
from tqdm import tqdm

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", help_string="Path to configuration file.")

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _abs_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))

# ------------------------------------------------------------------
# default config
# ------------------------------------------------------------------
def get_config():
    """
    Returns a configuration dictionary.
    NOTE: Batch size and sequence length have been reduced to safe starting values
    to prevent memory-related hangs. You can increase them gradually to find
    the maximum your hardware supports.
    """
    c = ConfigDict()
    c.model_path = _abs_path("2b-it/2b-it")
    c.tokenizer_path = _abs_path("2b-it/tokenizer.model")
    c.ckpt_dir = _abs_path("finetuning_checkpoints")
    c.learning_rate = 1e-5
    c.num_epochs = 3
    c.global_batch_size = 4      # REDUCED from 32 to prevent memory hang
    c.grad_clip_norm = 1.0
    c.grad_accum_steps = 1
    c.eval_steps = 250
    c.eval_batch_size = 4        # REDUCED from x32
    c.dataset_name = "FrenzyMath/Herald_proofs"
    c.train_split = "train"
    c.eval_split = "valid"
    c.max_seq_len = 1024         # REDUCED from 2048
    c.dataset_fraction = 0.001
    c.weight_dtype = jnp.bfloat16
    c.data_axis = "data"
    c.model_axis = "model"
    return c

# ------------------------------------------------------------------
# dataset
# ------------------------------------------------------------------
def create_dataset_iterator(config, split, mesh, batch_size):
    tokenizer = spm.SentencePieceProcessor(model_file=config.tokenizer_path)
    per_host = batch_size // jax.process_count()

    def gen():
        ds = load_dataset(config.dataset_name, split=split, streaming=True)
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
        if yielded == 0:
            yield [tokenizer.bos_id(), tokenizer.eos_id()]

    tf_ds = (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        .shard(num_shards=jax.process_count(), index=jax.process_index())
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

# ------------------------------------------------------------------
# model
# ------------------------------------------------------------------
def get_partition_rules():
    return (
        ("embedder/input_embedding", PartitionSpec("model", None)),
        ("readout/kernel", PartitionSpec(None, "model")),
        ("mlp_block/ffw_up/w", PartitionSpec(None, None, "model")),
        ("mlp_block/ffw_down/kernel", PartitionSpec("model", None)),
        ("attention_block/proj_q/kernel", PartitionSpec(None, "model")),
        ("attention_block/proj_final/kernel", PartitionSpec("model", None)),
        (r".*kernel", PartitionSpec("data", None)),
        (r".*w", PartitionSpec("data", None)),
        (r".*bias|.*b|.*scale|.*a_param", PartitionSpec()),
    )

def load_and_shard_model(config, mesh):
    # Initialize on the default device, not forcing CPU

    model_cfg.scan_backend = 'lax'
    model_cfg = rg.GriffinConfig.from_preset(rg.Preset.RECURRENT_GEMMA_2B_V1)
    model = rg.Griffin(model_cfg, dtype=config.weight_dtype)
    params_cpu = ocp.PyTreeCheckpointer().restore(config.model_path)

    try:
        from flax.training.common_utils import get_logical_partition_rules
        pspec_tree = get_logical_partition_rules(params_cpu, get_partition_rules())
    except ImportError:
        logging.warning("Using basic sharding – upgrade flax for logical rules")
        pspec_tree = jtu.tree_map(lambda _: PartitionSpec(), params_cpu)

    shardings = jtu.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), pspec_tree)
    with mesh:
        params_sharded = jax.device_put(params_cpu, shardings)
    return model, params_sharded, pspec_tree

# ------------------------------------------------------------------
# loss / step
# ------------------------------------------------------------------
class TrainState(train_state.TrainState):
    pass

def make_sharded_forward_fn(model, mesh, data_axis):
    """
    Return a jit-able, TPU-sharded forward+loss function for RecurrentGemma.
    This version correctly aligns inputs and targets BEFORE the model call.
    """
    def _forward(tokens, positions, params):
        # tokens: [B, L]
        # positions: [B, L]

        # 1. Build inputs and targets BEFORE the model call.
        # The model's task is to predict the next token.
        inputs  = tokens[:, :-1]          # Shape: [B, L-1]
        targets = tokens[:, 1:]           # Shape: [B, L-1]

        # 2. Forward pass with the correctly sized input
        logits, _ = model.apply(
            {"params": params},
            tokens=inputs,
            segment_pos=positions[:, :-1],  # Match the input's length
            cache=None,
        )                                  # Logits will have shape [B, L-1, V]

        # 3. Flatten for optax loss calculation
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)

        # 4. Mask out padding tokens (where target is 0)
        mask = targets_flat != 0

        # 5. Calculate cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits_flat.astype(jnp.float32),
            targets_flat
        )
        loss = jnp.sum(loss * mask) / jnp.maximum(mask.sum(), 1e-8)
        return loss

    # Shard the function across the data axis of the mesh
    sharded = shard_map(
        _forward,
        mesh=mesh,
        in_specs=(
            PartitionSpec(data_axis),  # tokens
            PartitionSpec(data_axis),  # positions
            PartitionSpec(),           # params (already sharded)
        ),
        out_specs=PartitionSpec(),     # scalar loss per shard
        check_rep=False,
    )
    return sharded


@partial(
    jax.jit,
    static_argnames=["model", "optimizer", "mesh", "data_axis"],
    donate_argnames=["params", "opt_state"],
)
def train_step(
    model,
    params,
    optimizer,
    opt_state,
    batch,
    *,
    mesh,
    data_axis,
):
    fwd = make_sharded_forward_fn(model, mesh, data_axis)

    def _loss(p):
        positions = jnp.broadcast_to(
            jnp.arange(batch["inputs"].shape[1])[None, :], batch["inputs"].shape
        )
        return fwd(batch["inputs"], positions, p)

    loss, grads = jax.value_and_grad(_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, {"loss": loss}

@partial(jax.jit, static_argnames=["model", "mesh", "data_axis"])
def eval_step(model, params, batch, *, mesh, data_axis):
    fwd = make_sharded_forward_fn(model, mesh, data_axis)
    positions = jnp.broadcast_to(
        jnp.arange(batch["inputs"].shape[1])[None, :], batch["inputs"].shape
    )
    loss = fwd(batch["inputs"], positions, params)
    return {"loss": loss}

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    cfg = get_config()
    jax.distributed.initialize()
    tf.config.set_visible_devices([], "GPU")

    devices = np.array(jax.devices()).reshape(
        jax.process_count(), jax.local_device_count()
    )
    mesh = Mesh(devices, (cfg.data_axis, cfg.model_axis))

    logging.set_verbosity(logging.INFO)
    if jax.process_index() == 0:
        logging.info(f"Mesh {mesh} ready on {jax.process_count()} hosts")
        logging.info(f"Starting with config: batch_size={cfg.global_batch_size}, seq_len={cfg.max_seq_len}")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    train_it = create_dataset_iterator(cfg, cfg.train_split, mesh, cfg.global_batch_size)
    eval_it = create_dataset_iterator(cfg, cfg.eval_split, mesh, cfg.eval_batch_size)

    model, params, pspecs = load_and_shard_model(cfg, mesh)

    base_opt = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adafactor(learning_rate=cfg.learning_rate),
    )
    opt = optax.MultiSteps(base_opt, every_k_schedule=cfg.grad_accum_steps)
    opt_state = opt.init(params)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    ckpt_mgr = ocp.CheckpointManager(
        cfg.ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )
    best_eval = float("inf")

    for epoch in range(cfg.num_epochs):
        steps = cfg.eval_steps * 5
        if jax.process_index() == 0:
            pbar = tqdm(total=steps, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        train_losses = []
        for step in range(1, steps + 1):
            batch = next(train_it)

            params, opt_state, metrics = train_step(
                model, params, opt, opt_state, batch,
                mesh=mesh,
                data_axis=cfg.data_axis,
            )
            # Ensure step completes before continuing
            metrics['loss'].block_until_ready()
            train_losses.append(metrics["loss"])

            if jax.process_index() == 0:
                pbar.update(1)

            if step % cfg.eval_steps == 0:
                multihost_utils.sync_global_devices("eval_barrier")
                eval_losses = []
                for _ in range(50):
                    eval_batch = next(eval_it)
                    eval_metrics = eval_step(
                        model, params, eval_batch,
                        mesh=mesh,
                        data_axis=cfg.data_axis,
                    )
                    eval_metrics['loss'].block_until_ready()
                    eval_losses.append(eval_metrics["loss"])

                eval_loss = jnp.mean(jnp.array(eval_losses))
                avg_train = jnp.mean(jnp.array(train_losses))

                if jax.process_index() == 0:
                    pbar.set_postfix(train=float(avg_train), eval=float(eval_loss))
                    logging.info(
                        f"Step {step}: train={avg_train:.4f} eval={eval_loss:.4f}"
                    )
                    train_losses.clear()

                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        state = state.replace(params=params, opt_state=opt_state, step=step)
                        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
                        logging.info("Saved best checkpoint")

        if jax.process_index() == 0:
            pbar.close()

    ckpt_mgr.wait_until_finished()
    if jax.process_index() == 0:
        logging.info("✅ Training complete.")

if __name__ == "__main__":
    app.run(main)

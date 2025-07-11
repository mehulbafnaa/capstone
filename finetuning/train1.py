import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state  # Use the standard TrainState
from tqdm import tqdm
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit
import jax.tree_util
from datasets import load_dataset

from utils.model_loader import load_recurrent_gemma_model
from finetuning.data_pipeline import get_dataset
from finetuning.config import (
    CKPT_DIR,
    TOK_FILE,
    TRAIN_SPLIT,
    DATASET_NAME,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    # GRADIENT_ACCUMULATION_STEPS is no longer needed
    CHECKPOINT_DIR,
    WEIGHT_DTYPE,
    DATASET_PROPORTION,
)

# Using the standard flax TrainState as we no longer need to store accumulated gradients.
TrainState = train_state.TrainState

# Loss function remains the same, can be jitted for performance.
@jax.jit
def calculate_loss(logits, labels):
    """Calculates the cross-entropy loss between logits and labels."""
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss_mask = (labels_flat != -100)
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits_flat.astype(jnp.float32), labels=labels_flat
    )
    masked_losses = jnp.where(loss_mask, losses, 0.0)
    total_loss = jnp.sum(masked_losses)
    num_valid_tokens = jnp.sum(loss_mask)
    loss = total_loss / (num_valid_tokens + 1e-8)
    return loss

def train_step(state, batch, base_dropout_rng, step_num):
    """
    Performs a single training step.
    This function now calculates the loss, computes gradients, and applies them immediately.
    """
    step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

    def loss_fn(params):
        """The loss function to be differentiated."""
        logits = state.apply_fn(
            {"params": params},
            tokens=batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": step_dropout_key}
        )[0]
        return calculate_loss(logits, batch["labels"])

    # Compute both the loss and the gradients in a single pass.
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Apply the gradients to update the model parameters.
    state = state.apply_gradients(grads=grads)
    
    return state, loss

def print_training_summary(num_devices, effective_batch_size, raw_dataset_size):
    """Prints a summary of the training configuration."""
    global_batch_size = effective_batch_size * num_devices
    
    examples_to_use = int(raw_dataset_size * DATASET_PROPORTION)
    num_train_steps_per_epoch = examples_to_use // global_batch_size

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION SUMMARY (No Gradient Accumulation)")
    print("="*80)
    print(f"Number of accelerator devices: {num_devices}")
    print(f"Per-device batch size: {effective_batch_size}")
    print(f"Global batch size (per optimizer step): {global_batch_size}")
    print(f"Raw dataset size: {raw_dataset_size}")
    print(f"Proportion to use: {DATASET_PROPORTION * 100:.1f}% ({examples_to_use} examples)")
    print(f"Optimizer steps per epoch: {num_train_steps_per_epoch}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("="*80 + "\n")


def setup(mesh):
    """Handles all the boilerplate setup for model, optimizer, and state."""
    if jax.process_index() == 0:
        print("Setting up model, optimizer, and sharded training state...")

    data_sharding = NamedSharding(mesh, PartitionSpec('data_axis'))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def get_param_sharding(param_pytree):
        """Defines sharding rules for model parameters."""
        def get_spec(param):
            if param.ndim > 1 and param.size > 1_000_000:
                return PartitionSpec(*([None] * (param.ndim - 1) + ['data_axis']))
            return PartitionSpec()
        return jax.tree.map(get_spec, param_pytree)

    model, _, params, _ = load_recurrent_gemma_model(
        CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE
    )
    
    class ScanShardingHelper:
        def __init__(self, mesh):
            self.mesh = mesh
            self.sequence_axis_name = None
            self.sequence_axis_index_groups = None
            self.activations_sharding_spec = PartitionSpec('data_axis')
            self.rnn_state_sharding_spec = PartitionSpec('data_axis')

    model.scan_sharding_spec = ScanShardingHelper(mesh=mesh)

    param_sharding_rules = get_param_sharding(params)

    optimizer = optax.adafactor(learning_rate=LEARNING_RATE)
    dummy_opt_state = optimizer.init(params)
    if isinstance(dummy_opt_state, tuple):
        opt_state_sharding_rules = tuple(get_param_sharding(s) for s in dummy_opt_state)
    else:
        opt_state_sharding_rules = get_param_sharding(dummy_opt_state)

    # Sharding specification for the standard TrainState.
    state_sharding_spec = TrainState(
        step=PartitionSpec(),
        apply_fn=None,
        params=param_sharding_rules,
        tx=None,
        opt_state=opt_state_sharding_rules,
    )

    # Create TrainState without pjit first, then shard it
    train_state = TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=optimizer
    )
    
    # Now shard the created state
    p_train_state = jax.device_put(train_state, NamedSharding(mesh, state_sharding_spec))
    
    del params, dummy_opt_state

    # pjit-compile the unified training step.
    p_train_step = pjit(
        train_step,
        in_shardings=(state_sharding_spec, data_sharding, replicated_sharding, replicated_sharding),
        out_shardings=(state_sharding_spec, replicated_sharding),
        donate_argnums=(0,) # Donate the state buffer for in-place update.
    )

    return p_train_state, p_train_step

def run_training_loop(state, p_train_step, train_dataset):
    """Executes the main training loop over epochs and steps."""
    if jax.process_index() == 0:
        print("Starting training...")

    rng = jax.random.PRNGKey(0)
    base_dropout_rng = jax.random.fold_in(rng, jax.process_index())
    
    num_train_steps_per_epoch = len(train_dataset)

    for epoch in range(NUM_EPOCHS):
        if jax.process_index() == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            pbar = tqdm(train_dataset, total=num_train_steps_per_epoch, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataset

        for step, batch in enumerate(pbar):
            batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
            
            # Perform a full training step (forward, backward, and optimizer update).
            state, loss = p_train_step(state, batch, base_dropout_rng, step)

            if jax.process_index() == 0:
                # Update the progress bar with the loss from the current step.
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
    
    return state

def main():
    """Main orchestration function."""
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}; Local devices: {jax.local_device_count()}; Global devices: {jax.device_count()}")

    num_devices = jax.device_count()
    
    effective_batch_size = BATCH_SIZE
    if BATCH_SIZE == 1:
        effective_batch_size = 2
        if jax.process_index() == 0:
            print("\nWARNING: BATCH_SIZE is 1, temporarily overriding to 2 to avoid library bug.\n")

    if jax.process_index() == 0:
        print("Loading dataset metadata to calculate training steps...")
        raw_dataset_for_size = load_dataset(DATASET_NAME, split=TRAIN_SPLIT)
        raw_dataset_size = len(raw_dataset_for_size)
        del raw_dataset_for_size
        print_training_summary(num_devices, effective_batch_size, raw_dataset_size)
    
    train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)

    with Mesh(jax.devices(), axis_names=('data_axis',)) as mesh:
        state, p_train_step = setup(mesh)
        final_state = run_training_loop(state, p_train_step, train_dataset)

        jax.block_until_ready(final_state)
        if jax.process_index() == 0:
            print("\nSaving final checkpoint...")
            ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
            raw_dataset_size = len(load_dataset(DATASET_NAME, split=TRAIN_SPLIT))
            num_train_steps_per_epoch = (int(raw_dataset_size * DATASET_PROPORTION) // (effective_batch_size * num_devices))
            final_step = num_train_steps_per_epoch * NUM_EPOCHS
            ckpt_manager.save(step=final_step, items=final_state)
            print("Final checkpoint saved.")

    if jax.process_index() == 0:
        print("\nTraining complete.")

if __name__ == "__main__":
    from finetuning.pretokenize_dataset import pretokenize_and_save
    pretokenize_and_save()
    main()

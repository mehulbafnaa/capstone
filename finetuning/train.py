
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from flax.core import freeze, unfreeze
from tqdm import tqdm
import time
from pathlib import Path

from utils.model_loader import load_recurrent_gemma_model
from finetuning.data_pipeline import get_dataset
from finetuning.config import (
    CKPT_DIR,
    TOK_FILE,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    MAX_SEQ_LEN,
    GRADIENT_ACCUMULATION_STEPS,
    CHECKPOINT_DIR,
    SAVE_EVERY_N_STEPS,
    EVAL_EVERY_N_STEPS,
    WEIGHT_DTYPE,
    ACTIVATION_DTYPE,
    DATASET_PROPORTION,
)

# Ensure JAX is initialized for distributed training
# jax.distributed.initialize() # Removed as it might be redundant or cause issues in some TPU environments

class TrainState(train_state.TrainState):
    accum_grads: any

@jax.jit
def calculate_loss(logits, labels):
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss_mask = (labels_flat != -100)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat.astype(jnp.float32), labels=labels_flat)
    masked_losses = jnp.where(loss_mask, losses, 0.0)
    total_loss = jnp.sum(masked_losses)
    num_valid_tokens = jnp.sum(loss_mask)
    loss = total_loss / (num_valid_tokens + 1e-8)
    return loss

def train_step(state, batch, dropout_rng):
    """
    Performs a single training step, including gradient accumulation.
    """
    # Create a new dropout key for each step
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": new_dropout_rng}
        )[0]
        loss = calculate_loss(logits, batch["labels"])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Accumulate gradients
    state = state.replace(
        accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads)
    )
    return state, loss, new_dropout_rng

@jax.jit
def apply_accumulated_gradients(state):
    """
    Applies the accumulated gradients to update the model parameters.
    """
    # Average gradients across devices and accumulation steps
    avg_grads = jax.lax.pmean(state.accum_grads, axis_name="batch")
    avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, avg_grads)

    # Update state
    state = state.apply_gradients(grads=avg_grads)

    # Reset accumulated gradients
    state = state.replace(
        accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads)
    )
    return state

def main():
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Global devices: {jax.device_count()}")

    # Load model with specified precision
    model, _, params, _ = load_recurrent_gemma_model(
        CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE
    )

    rng = jax.random.PRNGKey(0)
    rng = jax.random.fold_in(rng, jax.process_index())
    dropout_rng = jax.random.fold_in(rng, jax.process_index() + 1)

    def create_train_state_local(rng):
        """Creates an initial `TrainState` with mixed precision and gradient accumulation state."""
        # Initialize parameters with the correct dtype
        params = model.init(rng, jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32), jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32))["params"]
        params = jax.tree.map(lambda x: x.astype(WEIGHT_DTYPE), params)

        # Create optimizer with bfloat16 master weights
        optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=0.0)
        
        # Initialize accumulated gradients with zeros
        accum_grads = jax.tree.map(jnp.zeros_like, params)

        return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, accum_grads=accum_grads)

    # pmap the creation of the train state
    p_create_train_state = jax.pmap(create_train_state_local, axis_name="batch")
    p_train_state = p_create_train_state(jax.random.split(rng, jax.local_device_count()))

    ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR)

    # Load the pre-tokenized dataset
    train_dataset = get_dataset(TRAIN_SPLIT, BATCH_SIZE)

    # Get the full dataset length
    try:
        full_dataset_size = len(train_dataset)
    except TypeError:
        if jax.process_index() == 0:
            print("Warning: The training dataset has no `__len__`. Cannot use DATASET_PROPORTION.")
        full_dataset_size = None

    # Take a subset of the dataset if DATASET_PROPORTION is set
    if full_dataset_size and DATASET_PROPORTION < 1.0:
        subset_size = int(full_dataset_size * DATASET_PROPORTION)
        train_dataset = train_dataset.take(subset_size)
        num_train_steps = subset_size
        if jax.process_index() == 0:
            print(f"Using {DATASET_PROPORTION*100:.2f}% of the dataset: {num_train_steps} steps.")
    else:
        num_train_steps = full_dataset_size

    if jax.process_index() == 0:
        print("Starting training with gradient accumulation and mixed precision...")

    p_train_step = jax.pmap(train_step, axis_name="batch", in_axes=(0, 0, None), out_axes=(0, 0, None))
    p_apply_grads = jax.pmap(apply_accumulated_gradients, axis_name="batch")

    for epoch in range(NUM_EPOCHS):
        if jax.process_index() == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataset

        total_loss = 0
        for step, batch in enumerate(pbar):
            batch = jax.tree.map(lambda x: x.numpy(), batch)
            batch = jax.tree.map(lambda x: jnp.array(x), batch)
            local_device_count = jax.local_device_count()
            batch = jax.tree.map(lambda x: x.reshape(local_device_count, x.shape[0] // local_device_count, *x.shape[1:]), batch)
            
            p_train_state, loss, dropout_rng = p_train_step(p_train_state, batch, dropout_rng)
            total_loss += loss.mean()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                p_train_state = p_apply_grads(p_train_state)
                
                if jax.process_index() == 0:
                    avg_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    total_loss = 0

                optimizer_step = (step + 1) // GRADIENT_ACCUMULATION_STEPS
                if jax.process_index() == 0 and optimizer_step > 0 and optimizer_step % SAVE_EVERY_N_STEPS == 0:
                    # Ensure gradients are applied before saving
                    unreplicated_state_for_save = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
                    ckpt_manager.save(step=optimizer_step, items=unreplicated_state_for_save)
                    print(f"Checkpoint saved at optimizer step {optimizer_step}")
                    del unreplicated_state_for_save # Free memory after saving

    if jax.process_index() == 0:
        print("Training complete.")
        # Ensure final gradients are applied before saving
        p_train_state = p_apply_grads(p_train_state)
        
        # More aggressive memory cleanup before final save
        del p_train_step, p_apply_grads
        if 'batch' in locals():
            del batch
        if 'loss' in locals():
            del loss
        jax.block_until_ready(p_train_state)
        jax.clear_caches()

        unreplicated_state = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
        if num_train_steps:
            final_step = num_train_steps // GRADIENT_ACCUMULATION_STEPS
            ckpt_manager.save(step=final_step, items=unreplicated_state)
        else:
            ckpt_manager.save(step="final", items=unreplicated_state)
        print("Final checkpoint saved.")

if __name__ == "__main__":
    # This will run the pre-tokenization if the data is not found
    from finetuning.pretokenize_dataset import pretokenize_and_save
    pretokenize_and_save()
    main()

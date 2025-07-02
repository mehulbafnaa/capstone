import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from tqdm import tqdm
import time
from pathlib import Path

from utils.model_loader import load_recurrent_gemma_model
from finetuning.data_pipeline import get_dataset
from finetuning.config import (
    CKPT_DIR,
    TOK_FILE,
    DATASET_NAME,
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
)

# Ensure JAX is initialized for distributed training
jax.distributed.initialize()

# Define a simple TrainState for managing training parameters
class TrainState(train_state.TrainState):
    # Add any additional state here if needed, e.g., PRNGKey for dropout
    pass

@jax.jit
def calculate_loss(logits, labels):
    """
    Calculates the cross-entropy loss, ignoring padded tokens (-100).
    """
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Create a mask to ignore -100 labels (for prompt tokens and padding)
    loss_mask = (labels_flat != -100)

    # Calculate loss for all tokens, then apply mask
    losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat, labels=labels_flat)
    
    # Apply the mask and average over valid tokens
    masked_losses = jnp.where(loss_mask, losses, 0.0)
    total_loss = jnp.sum(masked_losses)
    num_valid_tokens = jnp.sum(loss_mask)
    
    # Avoid division by zero if no valid tokens
    loss = total_loss / (num_valid_tokens + 1e-8)
    return loss

@jax.jit
def train_step(state, batch, dropout_rng):
    """
    Performs a single training step.
    """
    # Split the dropout_rng for each device
    dropout_rng = jax.random.fold_in(dropout_rng, jax.lax.axis_index('batch'))

    def loss_fn(params):
        # Model application with training=True for dropout etc.
        logits = state.apply_fn(
            {"params": params},
            batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": dropout_rng}
        )[0] # Assuming logits are the first element of the tuple
        
        # The labels are already correctly aligned with the logits.
        loss = calculate_loss(logits, batch["labels"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Global devices: {jax.device_count()}")

    model, vocab, params, _ = load_recurrent_gemma_model(CKPT_DIR, TOK_FILE)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.fold_in(rng, jax.process_index())
    dropout_rng = jax.random.fold_in(rng, jax.process_index() + 1)

    def create_train_state_local(rng, learning_rate, weight_decay=0.0):
        """Creates an initial `TrainState`."""
        params = model.init(rng, jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32), jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32))["params"]
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    p_train_state = jax.pmap(create_train_state_local, axis_name="batch", in_axes=(0, None))(jax.random.split(rng, jax.local_device_count()), LEARNING_RATE)

    ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR)

    train_dataset = get_dataset(DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE)

    if jax.process_index() == 0:
        print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        if jax.process_index() == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            pbar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataset

        for step, batch in enumerate(pbar):
            batch = jax.tree.map(lambda x: x.numpy(), batch) # Convert TF tensors to NumPy arrays
            batch = jax.tree.map(lambda x: jnp.array(x), batch) # Convert NumPy arrays to JAX arrays
            local_device_count = jax.local_device_count()
            batch = jax.tree.map(lambda x: x.reshape(local_device_count, x.shape[0] // local_device_count, *x.shape[1:]), batch)
            
            p_train_state, loss = jax.pmap(train_step, axis_name="batch", in_axes=(0, 0, None))(p_train_state, batch, dropout_rng)

            if jax.process_index() == 0:
                pbar.set_postfix(loss=f"{loss.mean():.4f}")

                if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                    ckpt_manager.save(step=step, items=p_train_state)
                    print(f"Checkpoint saved at step {step + 1}")

    if jax.process_index() == 0:
        print("Training complete.")
        ckpt_manager.save(step="final", items=p_train_state)
        print("Final checkpoint saved.")

if __name__ == "__main__":
    main()

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

def create_train_state(rng, model, learning_rate, weight_decay=0.0):
    """Creates an initial `TrainState`."""
    params = model.init(rng, jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32))["params"]
    
    # Define the optimizer
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

@jax.jit
def calculate_loss(logits, labels, attention_mask):
    """
    Calculates the cross-entropy loss, ignoring padded tokens and prompt tokens.
    """
    # Shift logits and labels for next-token prediction
    # logits: (batch_size, sequence_length, vocab_size)
    # labels: (batch_size, sequence_length)
    
    # For causal language modeling, we predict the next token.
    # So, logits for token i predict token i+1.
    # We align labels by shifting them.
    
    # Optax's softmax_cross_entropy_with_integer_labels expects logits and labels
    # where labels are the target indices.
    
    # We need to flatten the logits and labels for optax.
    # Reshape to (batch_size * sequence_length, vocab_size) and (batch_size * sequence_length)
    
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Create a mask to ignore -100 labels (for prompt tokens and padding)
    # and also ignore padding from attention_mask
    loss_mask = (labels_flat != -100) * (attention_mask.reshape(-1) == 1)

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
        # Pass dropout_rng to the model apply function if it uses dropout
        logits = state.apply_fn({"params": params}, batch["input_ids"], train=True, rngs={"dropout": dropout_rng}).logits
        loss = calculate_loss(logits, batch["labels"], batch["attention_mask"])
        return loss, logits # Return logits for potential metrics/debugging

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")

    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    # Initialize JAX for distributed training
    # This is already called at the top of the script, but good to re-iterate
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Global devices: {jax.device_count()}")

    # Load model and tokenizer
    # Only process 0 loads the model to avoid redundant memory usage
    model, vocab, params, _ = load_recurrent_gemma_model(CKPT_DIR, TOK_FILE)

    # Create initial TrainState
    rng = jax.random.PRNGKey(0)
    # Split rng for each device for pmap
    rng = jax.random.fold_in(rng, jax.process_index())
    dropout_rng = jax.random.fold_in(rng, jax.process_index() + 1) # Separate rng for dropout

    # Replicate the initial state across devices
    def create_train_state_local(rng, learning_rate, weight_decay=0.0):
        """Creates an initial `TrainState`."""
        params = model.init(rng, jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32))["params"]
        
        # Define the optimizer
        optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        
        return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    p_train_state = jax.pmap(create_train_state_local, axis_name="batch", in_axes=(0, None))(jax.random.split(rng, jax.local_device_count()), LEARNING_RATE)

    # Setup checkpoint manager
    ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR)

    # Load datasets
    train_dataset = get_dataset(DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE)
    # validation_dataset = get_dataset(DATASET_NAME, VALIDATION_SPLIT, BATCH_SIZE, shuffle=False)

    if jax.process_index() == 0:
        print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        if jax.process_index() == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            pbar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataset # No progress bar for other processes

        for step, batch in enumerate(pbar):
            # Replicate batch data for pmap
            batch = jax.tree_map(lambda x: jnp.array(x), batch)
            batch = jax.tree_map(lambda x: jax.lax.broadcast(x, (jax.local_device_count(),)), batch)
            
            p_train_state, loss = jax.pmap(train_step, axis_name="batch")(p_train_state, batch, jax.random.split(dropout_rng, jax.local_device_count()))

            if jax.process_index() == 0:
                # Log loss from the first device (they should be identical after pmean)
                pbar.set_postfix(loss=f"{loss.mean():.4f}")

                # Checkpointing
                if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                    ckpt_manager.save(step=step, items=p_train_state)
                    print(f"Checkpoint saved at step {step + 1}")

                # Evaluation (simplified for now)
                # if (step + 1) % EVAL_EVERY_N_STEPS == 0:
                #     print(f"\nRunning evaluation at step {step + 1}...")
                #     # TODO: Implement evaluation logic using unified_eval.py or similar

    if jax.process_index() == 0:
        print("Training complete.")
        # Save final checkpoint
        ckpt_manager.save(step="final", items=p_train_state)
        print("Final checkpoint saved.")

if __name__ == "__main__":
    main()

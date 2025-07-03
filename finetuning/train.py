
# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from tqdm import tqdm
# import time
# from pathlib import Path

# from utils.model_loader import load_recurrent_gemma_model
# from finetuning.data_pipeline import get_dataset
# from finetuning.config import (
#     CKPT_DIR,
#     TOK_FILE,
#     DATASET_NAME,
#     TRAIN_SPLIT,
#     VALIDATION_SPLIT,
#     LEARNING_RATE,
#     BATCH_SIZE,
#     NUM_EPOCHS,
#     MAX_SEQ_LEN,
#     GRADIENT_ACCUMULATION_STEPS,
#     CHECKPOINT_DIR,
#     SAVE_EVERY_N_STEPS,
#     EVAL_EVERY_N_STEPS,
#     DATASET_PROPORTION, # New config for training on a subset
# )

# # Ensure JAX is initialized for distributed training
# # jax.distributed.initialize()

# # Define a simple TrainState for managing training parameters
# class TrainState(train_state.TrainState):
#     # Add any additional state here if needed, e.g., PRNGKey for dropout
#     pass

# def calculate_loss(logits, labels):
#     """
#     Calculates the cross-entropy loss, ignoring padded tokens (-100).
#     """
#     vocab_size = logits.shape[-1]
#     logits_flat = logits.reshape(-1, vocab_size)
#     labels_flat = labels.reshape(-1)

#     # Create a mask to ignore -100 labels (for prompt tokens and padding)
#     loss_mask = (labels_flat != -100)

#     # Calculate loss for all tokens, then apply mask
#     losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat, labels=labels_flat)
    
#     # Apply the mask and average over valid tokens
#     masked_losses = jnp.where(loss_mask, losses, 0.0)
#     total_loss = jnp.sum(masked_losses)
#     num_valid_tokens = jnp.sum(loss_mask)
    
#     # Avoid division by zero if no valid tokens
#     loss = total_loss / (num_valid_tokens + 1e-8)
#     return loss

# def grad_step(state, batch, dropout_rng):
#     """
#     Performs a forward and backward pass for a single micro-batch,
#     returning the gradients and loss.
#     """
#     def loss_fn(params):
#         # Model application with training=True for dropout etc.
#         logits = state.apply_fn(
#             {"params": params},
#             batch["input_ids"],
#             segment_pos=batch["segment_pos"],
#             rngs={"dropout": dropout_rng}
#         )[0] # Assuming logits are the first element of the tuple
        
#         # The labels are already correctly aligned with the logits.
#         loss = calculate_loss(logits, batch["labels"])
#         return loss, logits

#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (loss, _), grads = grad_fn(state.params)
    
#     # Return raw gradients and loss. They will be averaged and all-reduced later.
#     return grads, loss

# def main():
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Total processes: {jax.process_count()}")
#         print(f"Local devices: {jax.local_device_count()}")
#         print(f"Global devices: {jax.device_count()}")
#         if GRADIENT_ACCUMULATION_STEPS > 1:
#             print(f"Gradient accumulation enabled with {GRADIENT_ACCUMULATION_STEPS} steps.")

#     # Load the model and the pre-trained parameters
#     model, vocab, params, _ = load_recurrent_gemma_model(CKPT_DIR, TOK_FILE)

#     # Set up PRNG keys
#     rng = jax.random.PRNGKey(0)
#     rng = jax.random.fold_in(rng, jax.process_index())
#     dropout_rng = jax.random.fold_in(rng, jax.process_index() + 1)

#     # Create the optimizer
#     optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=0.0)
    
#     # Create the TrainState on the host CPU using the loaded parameters
#     state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
#     # Replicate the state to all local devices
#     p_train_state = jax.device_put_replicated(state, jax.local_devices())

#     ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR)

#     train_dataset = get_dataset(DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE)
    
#     # Get the full dataset length
#     try:
#         full_dataset_size = len(train_dataset)
#     except TypeError:
#         if jax.process_index() == 0:
#             print("Warning: The training dataset has no `__len__`. Cannot use DATASET_PROPORTION.")
#         full_dataset_size = None

#     # Take a subset of the dataset if DATASET_PROPORTION is set
#     if full_dataset_size and DATASET_PROPORTION < 1.0:
#         subset_size = int(full_dataset_size * DATASET_PROPORTION)
#         train_dataset = train_dataset.take(subset_size)
#         num_train_steps = subset_size
#         if jax.process_index() == 0:
#             print(f"Using {DATASET_PROPORTION*100:.2f}% of the dataset: {num_train_steps} steps.")
#     else:
#         num_train_steps = full_dataset_size

#     # pmap the gradient calculation step
#     p_grad_step = jax.pmap(grad_step, axis_name="batch", in_axes=(0, 0, None))

#     if jax.process_index() == 0:
#         print("Starting training...")

#     # The outer loop iterates over epochs.
#     for epoch in range(NUM_EPOCHS):
#         if jax.process_index() == 0:
#             print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#             # The progress bar now tracks micro-batches
#             pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
#         else:
#             pbar = train_dataset

#         # Initialize gradient and loss accumulators for the epoch
#         grad_accumulator = None
#         loss_accumulator = jax.device_put_replicated(0.0, jax.local_devices())
        
#         # Add a flag to show the batch warning only once
#         skipped_batch_warning_logged = False
        
#         # The inner loop iterates over micro-batches.
#         for step, batch in enumerate(pbar):
#             # Convert TF Tensors to JAX arrays
#             batch = jax.tree.map(lambda x: jnp.array(x.numpy()), batch)
            
#             local_device_count = jax.local_device_count()
#             current_batch_size = batch["input_ids"].shape[0]
            
#             # Skip batch if it cannot be evenly split, with a one-time warning
#             if current_batch_size % local_device_count != 0:
#                 if not skipped_batch_warning_logged and jax.process_index() == 0:
#                     print(f"\nWARNING: Skipping batches that cannot be evenly split across {local_device_count} devices (e.g., batch of size {current_batch_size}).")
#                     print("This is normal for the last batch, but if training is slow, ensure BATCH_SIZE in your config is a multiple of device count.")
#                     skipped_batch_warning_logged = True
#                 continue
            
#             # Reshape batch for distribution across local devices
#             batch = jax.tree.map(lambda x: x.reshape(local_device_count, x.shape[0] // local_device_count, *x.shape[1:]), batch)
            
#             # Calculate gradients for the current micro-batch
#             grads, loss = p_grad_step(p_train_state, batch, dropout_rng)

#             # Lazily initialize and accumulate gradients
#             if grad_accumulator is None:
#                 grad_accumulator = grads
#             else:
#                 grad_accumulator = jax.tree.map(lambda acc, g: acc + g, grad_accumulator, grads)
            
#             # Accumulate loss
#             loss_accumulator += loss

#             # An "optimizer step" occurs every GRADIENT_ACCUMULATION_STEPS
#             if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                 # Average gradients across accumulation steps
#                 final_grads = jax.tree.map(lambda g: g / GRADIENT_ACCUMULATION_STEPS, grad_accumulator)
#                 # All-reduce gradients across devices
#                 final_grads = jax.lax.pmean(final_grads, axis_name="batch")

#                 # Apply gradients to update the model state
#                 p_train_state = p_train_state.apply_gradients(grads=final_grads)

#                 # Reset gradient accumulator for the next cycle
#                 grad_accumulator = None

#                 # Log the loss
#                 if jax.process_index() == 0:
#                     # Average loss across accumulation steps and all-reduce across devices
#                     avg_loss = loss_accumulator / GRADIENT_ACCUMULATION_STEPS
#                     final_loss = jax.device_get(avg_loss)[0]
#                     pbar.set_postfix(loss=f"{final_loss:.4f}")
#                     # Reset loss accumulator
#                     loss_accumulator = jax.device_put_replicated(0.0, jax.local_devices())

#                 # Checkpointing logic (now based on optimizer steps)
#                 optimizer_step = (step + 1) // GRADIENT_ACCUMULATION_STEPS
#                 if jax.process_index() == 0 and optimizer_step > 0 and optimizer_step % SAVE_EVERY_N_STEPS == 0:
#                     unreplicated_state = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
#                     ckpt_manager.save(step=optimizer_step, items=unreplicated_state)
#                     print(f"Checkpoint saved at optimizer step {optimizer_step}")

#     if jax.process_index() == 0:
#         print("Training complete.")
        
#         # *** FIX: Clear device memory before saving the final model ***
#         # Delete large objects that are no longer needed to free up HBM
#         del p_grad_step
#         if 'grad_accumulator' in locals():
#             del grad_accumulator
#         if 'loss_accumulator' in locals():
#             del loss_accumulator
#         # Clear JAX's internal caches to release memory
#         jax.clear_caches()

#         # Save the final unreplicated state
#         unreplicated_state = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
#         if num_train_steps:
#             final_step = num_train_steps // GRADIENT_ACCUMULATION_STEPS
#             ckpt_manager.save(step=final_step, items=unreplicated_state)
#         else:
#             ckpt_manager.save(step="final", items=unreplicated_state)
#         print("Final checkpoint saved.")

# if __name__ == "__main__":
#     main()

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
    DATASET_PROPORTION, # New config for training on a subset
)

# # Ensure JAX is initialized for distributed training
# jax.distributed.initialize()

# Define a simple TrainState for managing training parameters
class TrainState(train_state.TrainState):
    # Add any additional state here if needed, e.g., PRNGKey for dropout
    pass

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

def grad_step(state, batch, dropout_rng):
    """
    Performs a forward and backward pass for a single micro-batch,
    returning the gradients and loss.
    """
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
    (loss, _), grads = grad_fn(state.params)
    
    # Return raw gradients and loss. They will be averaged and all-reduced later.
    return grads, loss

def main():
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Global devices: {jax.device_count()}")
        if GRADIENT_ACCUMULATION_STEPS > 1:
            print(f"Gradient accumulation enabled with {GRADIENT_ACCUMULATION_STEPS} steps.")

    # Load the model and the pre-trained parameters
    model, vocab, params, _ = load_recurrent_gemma_model(CKPT_DIR, TOK_FILE)

    # Set up PRNG keys
    rng = jax.random.PRNGKey(0)
    rng = jax.random.fold_in(rng, jax.process_index())
    dropout_rng = jax.random.fold_in(rng, jax.process_index() + 1)

    # Create the optimizer
    optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=0.0)
    
    # Create the TrainState on the host CPU using the loaded parameters
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    # Replicate the state to all local devices
    p_train_state = jax.device_put_replicated(state, jax.local_devices())

    ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR)

    train_dataset = get_dataset(DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE)
    
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

    # pmap the gradient calculation step
    p_grad_step = jax.pmap(grad_step, axis_name="batch", in_axes=(0, 0, None))

    if jax.process_index() == 0:
        print("Starting training...")

    # The outer loop iterates over epochs.
    for epoch in range(NUM_EPOCHS):
        if jax.process_index() == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_dataset

        grad_accumulator = None
        loss_accumulator = jax.device_put_replicated(0.0, jax.local_devices())
        skipped_batch_warning_logged = False
        
        for step, batch in enumerate(pbar):
            batch = jax.tree.map(lambda x: jnp.array(x.numpy()), batch)
            
            local_device_count = jax.local_device_count()
            current_batch_size = batch["input_ids"].shape[0]
            
            if current_batch_size % local_device_count != 0:
                if not skipped_batch_warning_logged and jax.process_index() == 0:
                    print(f"\nWARNING: Skipping batches that cannot be evenly split across {local_device_count} devices.")
                    skipped_batch_warning_logged = True
                continue
            
            batch = jax.tree.map(lambda x: x.reshape(local_device_count, -1, *x.shape[1:]), batch)
            
            grads, loss = p_grad_step(p_train_state, batch, dropout_rng)

            if grad_accumulator is None:
                grad_accumulator = grads
            else:
                grad_accumulator = jax.tree.map(lambda acc, g: acc + g, grad_accumulator, grads)
            
            loss_accumulator += loss

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                final_grads = jax.tree.map(lambda g: g / GRADIENT_ACCUMULATION_STEPS, grad_accumulator)
                final_grads = jax.lax.pmean(final_grads, axis_name="batch")
                p_train_state = p_train_state.apply_gradients(grads=final_grads)
                grad_accumulator = None

                if jax.process_index() == 0:
                    avg_loss = loss_accumulator / GRADIENT_ACCUMULATION_STEPS
                    final_loss = jax.device_get(avg_loss)[0]
                    pbar.set_postfix(loss=f"{final_loss:.4f}")
                    loss_accumulator = jax.device_put_replicated(0.0, jax.local_devices())

                optimizer_step = (step + 1) // GRADIENT_ACCUMULATION_STEPS
                if jax.process_index() == 0 and optimizer_step > 0 and optimizer_step % SAVE_EVERY_N_STEPS == 0:
                    # During training, we might need to be careful with memory for saving too.
                    # This approach gets one copy, which should be fine for intermediate saves.
                    unreplicated_state_for_save = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
                    ckpt_manager.save(step=optimizer_step, items=unreplicated_state_for_save)
                    print(f"Checkpoint saved at optimizer step {optimizer_step}")
                    del unreplicated_state_for_save # Free memory after saving

    if jax.process_index() == 0:
        print("Training complete.")
        
        # *** FIX: More aggressive memory cleanup before final save ***
        # Delete all large, unnecessary variables from the training loop scope.
        del p_grad_step
        if 'grad_accumulator' in locals() and grad_accumulator is not None:
            del grad_accumulator
        if 'loss_accumulator' in locals():
            del loss_accumulator
        if 'batch' in locals():
            del batch
        if 'grads' in locals():
            del grads
        if 'loss' in locals():
            del loss
        if 'final_grads' in locals():
            del final_grads

        # This is a crucial step: ensure all pending operations on the devices are
        # finished before we try to clear memory and copy the final state.
        jax.block_until_ready(p_train_state)
        
        # Clear JAX's internal caches to release as much memory as possible.
        jax.clear_caches()

        # Save the final unreplicated state
        unreplicated_state = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
        if num_train_steps:
            final_step = num_train_steps // GRADIENT_ACCUMULATION_STEPS
            ckpt_manager.save(step=final_step, items=unreplicated_state)
        else:
            ckpt_manager.save(step="final", items=unreplicated_state)
        print("Final checkpoint saved.")

if __name__ == "__main__":
    main()

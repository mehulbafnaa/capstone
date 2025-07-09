
# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from flax.core import freeze, unfreeze
# from tqdm import tqdm
# import time
# from pathlib import Path

# from utils.model_loader import load_recurrent_gemma_model
# from finetuning.data_pipeline import get_dataset
# from finetuning.config import (
#     CKPT_DIR,
#     TOK_FILE,
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
#     WEIGHT_DTYPE,
#     ACTIVATION_DTYPE,
#     DATASET_PROPORTION,
# )

# class TrainState(train_state.TrainState):
#     accum_grads: any

# @jax.jit
# def calculate_loss(logits, labels):
#     vocab_size = logits.shape[-1]
#     logits_flat = logits.reshape(-1, vocab_size)
#     labels_flat = labels.reshape(-1)
#     loss_mask = (labels_flat != -100)
#     losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat.astype(jnp.float32), labels=labels_flat)
#     masked_losses = jnp.where(loss_mask, losses, 0.0)
#     total_loss = jnp.sum(masked_losses)
#     num_valid_tokens = jnp.sum(loss_mask)
#     loss = total_loss / (num_valid_tokens + 1e-8)
#     return loss

# def train_step(state, batch, dropout_rng):
#     """
#     Performs a single training step, including gradient accumulation.
#     """
#     # Create a new dropout key for each step
#     dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

#     def loss_fn(params):
#         logits = state.apply_fn(
#             {"params": params},
#             tokens=jnp.expand_dims(batch["input_ids"], axis=-1),
#             segment_pos=jnp.expand_dims(batch["segment_pos"], axis=-1),
#             rngs={"dropout": new_dropout_rng}
#         )[0]
#         loss = calculate_loss(logits, batch["labels"])
#         return loss

#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(state.params)
    
#     # Accumulate gradients
#     state = state.replace(
#         accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads)
#     )
#     return state, loss, new_dropout_rng

# @jax.jit
# def apply_accumulated_gradients(state):
#     """
#     Applies the accumulated gradients to update the model parameters.
#     """
#     # Average gradients across devices and accumulation steps
#     avg_grads = jax.lax.pmean(state.accum_grads, axis_name="batch")
#     avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, avg_grads)

#     # Update state
#     state = state.apply_gradients(grads=avg_grads)

#     # Reset accumulated gradients
#     state = state.replace(
#         accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads)
#     )
#     return state

# def main():
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Total processes: {jax.process_count()}")
#         print(f"Local devices: {jax.local_device_count()}")
#         print(f"Global devices: {jax.device_count()}")

#     # Load model with specified precision
#     model, _, params, _ = load_recurrent_gemma_model(
#         CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE
#     )

#     rng = jax.random.PRNGKey(0)
#     rng = jax.random.fold_in(rng, jax.process_index())
#     dropout_rng = jax.random.fold_in(rng, jax.process_index() + 1)

#     def create_train_state_local(rng):
#         """Creates an initial `TrainState` with mixed precision and gradient accumulation state."""
#         # Initialize parameters with the correct dtype
#         params = model.init(rng, jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32), jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32))["params"]
#         params = jax.tree.map(lambda x: x.astype(WEIGHT_DTYPE), params)

#         # Create optimizer with bfloat16 master weights
#         optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=0.0)
        
#         # Initialize accumulated gradients with zeros
#         accum_grads = jax.tree.map(jnp.zeros_like, params)

#         return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer, accum_grads=accum_grads)

#     # pmap the creation of the train state
#     p_create_train_state = jax.pmap(create_train_state_local, axis_name="batch")
#     p_train_state = p_create_train_state(jax.random.split(rng, jax.local_device_count()))

#     ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR)

#     # Load the pre-tokenized dataset
#     train_dataset = get_dataset(TRAIN_SPLIT, BATCH_SIZE * jax.local_device_count())

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

#     if jax.process_index() == 0:
#         print("Starting training with gradient accumulation and mixed precision...")

#     p_train_step = jax.pmap(train_step, axis_name="batch", in_axes=(0, 0, None), out_axes=(0, 0, None))
#     p_apply_grads = jax.pmap(apply_accumulated_gradients, axis_name="batch")

#     for epoch in range(NUM_EPOCHS):
#         if jax.process_index() == 0:
#             print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#             pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
#         else:
#             pbar = train_dataset

#         total_loss = 0
#         for step, batch in enumerate(pbar):
#             batch = jax.tree.map(lambda x: x.numpy(), batch)
#             batch = jax.tree.map(lambda x: jnp.array(x), batch)
#             local_device_count = jax.local_device_count()
#             # The batch from the dataset is already correctly shaped for pmap.
#             # No need to reshape here.
#             # batch = jax.tree.map(lambda x: x.reshape(local_device_count, x.shape[0] // local_device_count, *x.shape[1:]), batch)
            
#             p_train_state, loss, dropout_rng = p_train_step(p_train_state, batch, dropout_rng)
#             total_loss += loss.mean()

#             if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                 p_train_state = p_apply_grads(p_train_state)
                
#                 if jax.process_index() == 0:
#                     avg_loss = total_loss / GRADIENT_ACCUMULATION_STEPS
#                     pbar.set_postfix(loss=f"{avg_loss:.4f}")
#                     total_loss = 0

#                 optimizer_step = (step + 1) // GRADIENT_ACCUMULATION_STEPS
#                 if jax.process_index() == 0 and optimizer_step > 0 and optimizer_step % SAVE_EVERY_N_STEPS == 0:
#                     # Ensure gradients are applied before saving
#                     unreplicated_state_for_save = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
#                     ckpt_manager.save(step=optimizer_step, items=unreplicated_state_for_save)
#                     print(f"Checkpoint saved at optimizer step {optimizer_step}")
#                     del unreplicated_state_for_save # Free memory after saving

#     if jax.process_index() == 0:
#         print("Training complete.")
#         # Ensure final gradients are applied before saving
#         p_train_state = p_apply_grads(p_train_state)
        
#         # More aggressive memory cleanup before final save
#         del p_train_step, p_apply_grads
#         if 'batch' in locals():
#             del batch
#         if 'loss' in locals():
#             del loss
#         jax.block_until_ready(p_train_state)
#         jax.clear_caches()

#         unreplicated_state = jax.device_get(jax.tree.map(lambda x: x[0], p_train_state))
#         if num_train_steps:
#             final_step = num_train_steps // GRADIENT_ACCUMULATION_STEPS
#             ckpt_manager.save(step=final_step, items=unreplicated_state)
#         else:
#             ckpt_manager.save(step="final", items=unreplicated_state)
#         print("Final checkpoint saved.")

# if __name__ == "__main__":
#     # This will run the pre-tokenization if the data is not found
#     from finetuning.pretokenize_dataset import pretokenize_and_save
#     pretokenize_and_save()
#     main()



import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from flax.core import freeze, unfreeze
from tqdm import tqdm
import time
from pathlib import Path
# Import pjit and sharding utilities
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit
import jax.tree_util

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

# Custom TrainState to hold accumulated gradients
class TrainState(train_state.TrainState):
    accum_grads: any

# Loss function remains the same, can be jitted for performance
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

# --- The core training and gradient application steps will now be managed by pjit ---
def train_step(state, batch, base_dropout_rng, step_num):
    """
    Performs a single training step. RNG is handled by folding in the step number.
    """
    # This is the robust, idiomatic way to handle RNG with pjit.
    # Create a unique key for this specific training step by folding the step
    # number into the base RNG key.
    step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

    def loss_fn(params):
        # The model expects a 2D input of shape [batch, seq_len].
        logits = state.apply_fn(
            {"params": params},
            tokens=batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": step_dropout_key} # Use the unique key for this step
        )[0]
        loss = calculate_loss(logits, batch["labels"])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.replace(accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads))

    # We no longer need to thread the keys through the loop
    return state, loss

# def apply_accumulated_gradients(state):
#     """
#     Averages the accumulated gradients across all devices and applies the update.
#     This function is designed to be pjit-compiled.
#     """
#     # Average gradients across the 'data_axis' of the mesh
#     avg_grads = jax.lax.pmean(state.accum_grads, axis_name="data_axis")
#     # Scale gradients by the number of accumulation steps
#     avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, avg_grads)
#     # Apply the averaged gradients
#     state = state.apply_gradients(grads=avg_grads)
#     # Reset accumulated gradients to zero
#     state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
#     return state


def apply_accumulated_gradients(state):
    """
    Averages the accumulated gradients across all devices and applies the update.
    This function is designed to be pjit-compiled.
    
    Note: With pjit, we use psum instead of pmean because axis names 
    aren't automatically bound like they are with pmap.
    """
    # Get the number of devices for manual averaging
    device_count = jax.device_count()
    
    # Sum gradients across all devices using psum
    summed_grads = jax.lax.psum(state.accum_grads, axis_name="data_axis")
    
    # Manually compute the average
    avg_grads = jax.tree.map(lambda x: x / device_count, summed_grads)
    
    # Scale gradients by the number of accumulation steps
    avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, avg_grads)
    
    # Apply the averaged gradients
    state = state.apply_gradients(grads=avg_grads)
    
    # Reset accumulated gradients to zero
    state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
    
    return state

def main():
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Global devices: {jax.device_count()}")

    # All pjit operations must be within a Mesh context manager
    num_devices = jax.device_count()
    # Create a 1D mesh, where we will shard data and model parameters.
    with Mesh(jax.devices(), axis_names=('data_axis',)) as device_mesh:
        # --- FIX for library bug with batch size of 1 ---
        effective_batch_size = BATCH_SIZE
        if BATCH_SIZE == 1:
            effective_batch_size = 2
            if jax.process_index() == 0:
                print("\n" + "="*80)
                print("WARNING: The per-device BATCH_SIZE is set to 1.")
                print("The recurrentgemma library has a known issue handling a batch size of 1,")
                print("which can cause shape errors inside the model's layers.")
                print(f"Temporarily overriding per-device batch size to {effective_batch_size} to avoid this issue.")
                print("Please consider setting BATCH_SIZE > 1 in your config file for stable training.")
                print("="*80 + "\n")

        # Define sharding rules.
        data_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec('data_axis',))
        replicated_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec())

        def get_param_sharding(param_pytree):
            """A helper to define sharding rules for model parameters."""
            def get_spec(param):
                # Shard large parameters along the last dimension on the 'data_axis'
                if param.ndim > 1 and param.size > 1_000_000:
                    sharding_spec = [None] * (param.ndim - 1) + ['data_axis']
                    return PartitionSpec(*sharding_spec)
                else: # Replicate smaller parameters
                    return PartitionSpec()
            return jax.tree.map(get_spec, param_pytree)

        # 1. Load model and parameters on the CPU first.
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

        model.scan_sharding_spec = ScanShardingHelper(mesh=device_mesh)

        # 2. Create the optimizer ONCE on the host.
        optimizer = optax.adafactor(learning_rate=LEARNING_RATE)

        # 3. Define sharding for the ENTIRE TrainState.
        param_sharding_rules = get_param_sharding(params)
        dummy_opt_state = optimizer.init(params)

        if isinstance(dummy_opt_state, tuple):
            opt_state_sharding_rules = tuple(get_param_sharding(s) for s in dummy_opt_state)
        else:
            opt_state_sharding_rules = get_param_sharding(dummy_opt_state)

        state_sharding_spec = TrainState(
            step=PartitionSpec(),
            apply_fn=model.apply,
            params=param_sharding_rules,
            tx=optimizer,
            opt_state=opt_state_sharding_rules,
            accum_grads=param_sharding_rules
        )

        # 4. Define the function to create the sharded training state.
        def create_sharded_train_state(params):
            return TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optimizer,
                accum_grads=jax.tree.map(jnp.zeros_like, params)
            )

        # Compile the creation function, specifying sharding for inputs and outputs.
        p_create_sharded_train_state = pjit(
            create_sharded_train_state,
            in_shardings=(param_sharding_rules,),
            out_shardings=state_sharding_spec
        )
        # Run the creation function to get the distributed training state.
        p_train_state = p_create_sharded_train_state(params)
        del params, dummy_opt_state # Free CPU memory

        # --- pjit-compile the training functions with sharding info ---
        p_train_step = pjit(
            train_step,
            in_shardings=(state_sharding_spec, data_sharding, replicated_sharding, replicated_sharding),
            out_shardings=(state_sharding_spec, replicated_sharding)
        )
        p_apply_grads = pjit(
            apply_accumulated_gradients,
            in_shardings=(state_sharding_spec,),
            out_shardings=state_sharding_spec
        )

        # Initialize PRNG keys
        rng = jax.random.PRNGKey(0)
        base_dropout_rng = jax.random.fold_in(rng, jax.process_index())

        ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
        train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)

        try:
            num_train_steps = len(train_dataset) // (GRADIENT_ACCUMULATION_STEPS)
        except TypeError:
            num_train_steps = None

        if jax.process_index() == 0:
            print("Starting training with Model Parallelism (pjit)...")

        for epoch in range(NUM_EPOCHS):
            if jax.process_index() == 0:
                print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
                pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
            else:
                pbar = train_dataset

            total_loss = 0
            # We need to keep track of the global step for the progress bar
            global_step_counter = 0
            for step, batch in enumerate(pbar):
                batch = jax.tree.map(lambda x: x.numpy(), batch)
                sharded_batch = jax.device_put(batch, data_sharding)

                p_train_state, loss = p_train_step(p_train_state, sharded_batch, base_dropout_rng, step)
                total_loss += loss.mean()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    p_train_state = p_apply_grads(p_train_state)
                    if jax.process_index() == 0:
                        avg_loss = total_loss.item() / GRADIENT_ACCUMULATION_STEPS
                        pbar.set_postfix(loss=f"{avg_loss:.4f}")
                        pbar.update(1) # Manually update progress bar after each gradient step
                        total_loss = 0
                        global_step_counter +=1

            if jax.process_index() == 0 and not num_train_steps:
                 pbar.n = global_step_counter
                 pbar.total = global_step_counter
                 pbar.close()


        # =======================================================================
        # FIX: Apply any final gradients that didn't make a full accumulation batch.
        # This call MUST be inside the 'with Mesh(...)' block to have access
        # to the 'data_axis' name for the psum collective operation.
        # =======================================================================
        if jax.process_index() == 0:
            print("\nApplying final accumulated gradients...")
        
        # Apply final gradients. This is a collective operation.
        p_train_state = p_apply_grads(p_train_state)
        
        # Wait for all devices to finish before saving.
        jax.block_until_ready(p_train_state)

        # Save the final checkpoint from the main process.
        # Orbax's PyTreeCheckpointer can handle saving the sharded state.
        if jax.process_index() == 0:
            if num_train_steps:
                final_step = num_train_steps * NUM_EPOCHS
                ckpt_manager.save(step=final_step, items=p_train_state)
            else:
                ckpt_manager.save(step="final", items=p_train_state)
            print("Final checkpoint saved.")

        if jax.process_index() == 0:
            print("\nTraining complete.")

if __name__ == "__main__":
    from finetuning.pretokenize_dataset import pretokenize_and_save
    pretokenize_and_save()
    main()
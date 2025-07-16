

# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from flax.core import freeze, unfreeze
# from tqdm import tqdm
# import time
# from pathlib import Path
# # Import pjit and sharding utilities
# from jax.sharding import Mesh, PartitionSpec, NamedSharding
# from jax.experimental.pjit import pjit
# import jax.tree_util

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

# # Custom TrainState to hold accumulated gradients
# class TrainState(train_state.TrainState):
#     accum_grads: any

# # Loss function remains the same, can be jitted for performance
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

# # --- The core training and gradient application steps will now be managed by pjit ---
# def train_step(state, batch, base_dropout_rng, step_num):
#     """
#     Performs a single training step. RNG is handled by folding in the step number.
#     """
#     # This is the robust, idiomatic way to handle RNG with pjit.
#     # Create a unique key for this specific training step by folding the step
#     # number into the base RNG key.
#     step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

#     def loss_fn(params):
#         # The model expects a 2D input of shape [batch, seq_len].
#         logits = state.apply_fn(
#             {"params": params},
#             tokens=batch["input_ids"],
#             segment_pos=batch["segment_pos"],
#             rngs={"dropout": step_dropout_key} # Use the unique key for this step
#         )[0]
#         loss = calculate_loss(logits, batch["labels"])
#         return loss

#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(state.params)
#     state = state.replace(accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads))

#     # We no longer need to thread the keys through the loop
#     return state, loss



# def apply_accumulated_gradients(state):
#     """
#     Averages the accumulated gradients and applies the update.
#     This function is designed to be pjit-compiled.
#     """
#     # Simple approach - just scale by accumulation steps
#     # pjit handles cross-device communication automatically
#     avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
    
#     # Apply the gradients
#     state = state.apply_gradients(grads=avg_grads)
    
#     # Reset accumulated gradients to zero
#     state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
    
#     return state

# def main():
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Total processes: {jax.process_count()}")
#         print(f"Local devices: {jax.local_device_count()}")
#         print(f"Global devices: {jax.device_count()}")

#     # All pjit operations must be within a Mesh context manager
#     num_devices = jax.device_count()
#     # Create a 1D mesh, where we will shard data and model parameters.
#     with Mesh(jax.devices(), axis_names=('data_axis',)) as device_mesh:
#         # --- FIX for library bug with batch size of 1 ---
#         effective_batch_size = BATCH_SIZE
#         if BATCH_SIZE == 1:
#             effective_batch_size = 2
#             if jax.process_index() == 0:
#                 print("\n" + "="*80)
#                 print("WARNING: The per-device BATCH_SIZE is set to 1.")
#                 print("The recurrentgemma library has a known issue handling a batch size of 1,")
#                 print("which can cause shape errors inside the model's layers.")
#                 print(f"Temporarily overriding per-device batch size to {effective_batch_size} to avoid this issue.")
#                 print("Please consider setting BATCH_SIZE > 1 in your config file for stable training.")
#                 print("="*80 + "\n")

#         # Define sharding rules.
#         data_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec('data_axis',))
#         replicated_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec())

#         def get_param_sharding(param_pytree):
#             """A helper to define sharding rules for model parameters."""
#             def get_spec(param):
#                 # Shard large parameters along the last dimension on the 'data_axis'
#                 if param.ndim > 1 and param.size > 1_000_000:
#                     sharding_spec = [None] * (param.ndim - 1) + ['data_axis']
#                     return PartitionSpec(*sharding_spec)
#                 else: # Replicate smaller parameters
#                     return PartitionSpec()
#             return jax.tree.map(get_spec, param_pytree)

#         # 1. Load model and parameters on the CPU first.
#         model, _, params, _ = load_recurrent_gemma_model(
#             CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE
#         )

#         class ScanShardingHelper:
#             def __init__(self, mesh):
#                 self.mesh = mesh
#                 self.sequence_axis_name = None
#                 self.sequence_axis_index_groups = None
#                 self.activations_sharding_spec = PartitionSpec('data_axis')
#                 self.rnn_state_sharding_spec = PartitionSpec('data_axis')

#         model.scan_sharding_spec = ScanShardingHelper(mesh=device_mesh)

#         # 2. Create the optimizer ONCE on the host.
#         optimizer = optax.adafactor(learning_rate=LEARNING_RATE)

#         # 3. Define sharding for the ENTIRE TrainState.
#         param_sharding_rules = get_param_sharding(params)
#         dummy_opt_state = optimizer.init(params)

#         if isinstance(dummy_opt_state, tuple):
#             opt_state_sharding_rules = tuple(get_param_sharding(s) for s in dummy_opt_state)
#         else:
#             opt_state_sharding_rules = get_param_sharding(dummy_opt_state)

#         state_sharding_spec = TrainState(
#             step=PartitionSpec(),
#             apply_fn=model.apply,
#             params=param_sharding_rules,
#             tx=optimizer,
#             opt_state=opt_state_sharding_rules,
#             accum_grads=param_sharding_rules
#         )

#         # 4. Define the function to create the sharded training state.
#         def create_sharded_train_state(params):
#             return TrainState.create(
#                 apply_fn=model.apply,
#                 params=params,
#                 tx=optimizer,
#                 accum_grads=jax.tree.map(jnp.zeros_like, params)
#             )

#         # Compile the creation function, specifying sharding for inputs and outputs.
#         p_create_sharded_train_state = pjit(
#             create_sharded_train_state,
#             in_shardings=(param_sharding_rules,),
#             out_shardings=state_sharding_spec
#         )
#         # Run the creation function to get the distributed training state.
#         p_train_state = p_create_sharded_train_state(params)
#         del params, dummy_opt_state # Free CPU memory

#         # --- pjit-compile the training functions with sharding info ---
#         p_train_step = pjit(
#             train_step,
#             in_shardings=(state_sharding_spec, data_sharding, replicated_sharding, replicated_sharding),
#             out_shardings=(state_sharding_spec, replicated_sharding)
#         )
#         p_apply_grads = pjit(
#             apply_accumulated_gradients,
#             in_shardings=(state_sharding_spec,),
#             out_shardings=state_sharding_spec,
#             donate_argnums=(0,)
#         )

#         # Initialize PRNG keys
#         rng = jax.random.PRNGKey(0)
#         base_dropout_rng = jax.random.fold_in(rng, jax.process_index())

#         ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
#         train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)

#         try:
#             num_train_steps = len(train_dataset) // (GRADIENT_ACCUMULATION_STEPS)
#         except TypeError:
#             num_train_steps = None

#         if jax.process_index() == 0:
#             print("Starting training with Model Parallelism (pjit)...")

#         for epoch in range(NUM_EPOCHS):
#             if jax.process_index() == 0:
#                 print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#                 pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
#             else:
#                 pbar = train_dataset

#             total_loss = 0
#             global_step_counter = 0
            
#             for step, batch in enumerate(pbar):
#                 batch = jax.tree.map(lambda x: x.numpy(), batch)
#                 sharded_batch = jax.device_put(batch, data_sharding)

#                 p_train_state, loss = p_train_step(p_train_state, sharded_batch, base_dropout_rng, step)
#                 total_loss += loss.mean()

#                 if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#                     p_train_state = p_apply_grads(p_train_state)
#                     if jax.process_index() == 0:
#                         avg_loss = total_loss.item() / GRADIENT_ACCUMULATION_STEPS
#                         pbar.set_postfix(loss=f"{avg_loss:.4f}")
#                         pbar.update(1)
#                         total_loss = 0
#                         global_step_counter += 1

#             if jax.process_index() == 0 and not num_train_steps:
#                  pbar.n = global_step_counter
#                  pbar.total = global_step_counter
#                  pbar.close()

#         # CRITICAL FIX: Apply final gradients only if there are remaining accumulated gradients
#         # And ensure ALL processes participate in this collective operation
#         remaining_steps = step % GRADIENT_ACCUMULATION_STEPS
#         if remaining_steps > 0:
#             if jax.process_index() == 0:
#                 print(f"\nApplying final accumulated gradients ({remaining_steps} remaining steps)...")
            
#             # All processes must participate in this collective operation
#             p_train_state = p_apply_grads(p_train_state)
        
#         # # Ensure all processes synchronize before saving
#         # jax.block_until_ready(p_train_state)

#         # # Save the final checkpoint from the main process only
#         # if jax.process_index() == 0:
#         #     if num_train_steps:
#         #         final_step = num_train_steps * NUM_EPOCHS
#         #         ckpt_manager.save(step=final_step, items=p_train_state)
#         #     else:
#         #         ckpt_manager.save(step="final", items=p_train_state)
#         #     print("Final checkpoint saved.")

#         # if jax.process_index() == 0:
#         #     print("\nTraining complete.")
#         # Ensure all processes synchronize before saving
        
#         jax.block_until_ready(p_train_state)

#         # Save the final checkpoint from the main process only
#         if jax.process_index() == 0:
#             if num_train_steps:
#                 final_step = num_train_steps * NUM_EPOCHS
#                 ckpt_manager.save(step=final_step, items=p_train_state)
#             else:
#                 # This is the branch your code is likely taking
#                 ckpt_manager.save(step="final", items=p_train_state)
            
#             # Add this line to wait for the save to complete
#             ckpt_manager.wait_until_finished()

#             print("Final checkpoint saved and write-operation confirmed.")

#         if jax.process_index() == 0:
#             print("\nTraining complete.")


# if __name__ == "__main__":
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

def train_step(state, batch, base_dropout_rng, step_num):
    """
    Performs a single training step to accumulate gradients.
    """
    step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            tokens=batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": step_dropout_key}
        )[0]
        loss = calculate_loss(logits, batch["labels"])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.replace(accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads))
    return state, loss

def apply_accumulated_gradients(state):
    """
    Averages the accumulated gradients and applies the update.
    """
    avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
    state = state.apply_gradients(grads=avg_grads)
    state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
    return state

def main():
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Global devices: {jax.device_count()}")

    num_devices = jax.device_count()
    with Mesh(jax.devices(), axis_names=('data_axis',)) as device_mesh:
        effective_batch_size = BATCH_SIZE
        if BATCH_SIZE == 1:
            effective_batch_size = 2
            if jax.process_index() == 0:
                print("\n" + "="*80)
                print("WARNING: Temporarily overriding per-device BATCH_SIZE to 2 to avoid a library bug.")
                print("="*80 + "\n")

        data_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec('data_axis',))
        replicated_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec())

        def get_param_sharding(param_pytree):
            def get_spec(param):
                if param.ndim > 1 and param.size > 1_000_000:
                    sharding_spec = [None] * (param.ndim - 1) + ['data_axis']
                    return PartitionSpec(*sharding_spec)
                else:
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

        model.scan_sharding_spec = ScanShardingHelper(mesh=device_mesh)

        optimizer = optax.adafactor(learning_rate=LEARNING_RATE)
        param_sharding_rules = get_param_sharding(params)
        dummy_opt_state = optimizer.init(params)

        if isinstance(dummy_opt_state, tuple):
            opt_state_sharding_rules = tuple(get_param_sharding(s) for s in dummy_opt_state)
        else:
            opt_state_sharding_rules = get_param_sharding(dummy_opt_state)

        # This defines the sharding *rules* (PartitionSpec) for the state.
        state_sharding_spec = TrainState(
            step=PartitionSpec(),
            apply_fn=None,
            params=param_sharding_rules,
            tx=None,
            opt_state=opt_state_sharding_rules,
            accum_grads=param_sharding_rules
        )

        # Create the initial TrainState on the host CPU.
        if jax.process_index() == 0:
            print("Creating initial training state on CPU...")
        train_state_on_cpu = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            accum_grads=jax.tree.map(jnp.zeros_like, params)
        )
        del params, dummy_opt_state

        # --- REFACTORED BLOCK: Convert PartitionSpec rules to concrete NamedSharding objects ---
        # This explicitly tells device_put which mesh to use for the sharding rules.
        sharding_for_put = jax.tree.map(
            lambda spec: NamedSharding(device_mesh, spec),
            state_sharding_spec
        )

        if jax.process_index() == 0:
            print("Sharding state across all devices...")
        p_train_state = jax.device_put(train_state_on_cpu, sharding_for_put)
        # --- END OF REFACTORED BLOCK ---

        p_train_step = pjit(
            train_step,
            in_shardings=(state_sharding_spec, data_sharding, replicated_sharding, replicated_sharding),
            out_shardings=(state_sharding_spec, replicated_sharding)
        )
        p_apply_grads = pjit(
            apply_accumulated_gradients,
            in_shardings=(state_sharding_spec,),
            out_shardings=state_sharding_spec,
            donate_argnums=(0,)
        )

        rng = jax.random.PRNGKey(0)
        base_dropout_rng = jax.random.fold_in(rng, jax.process_index())
        ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
        train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)
        
        try:
            num_train_steps = len(train_dataset) // GRADIENT_ACCUMULATION_STEPS
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
                        if num_train_steps is not None:
                            pbar.update(1)
                        total_loss = 0
                        global_step_counter += 1

            if jax.process_index() == 0 and num_train_steps is None:
                 pbar.n = global_step_counter
                 pbar.total = global_step_counter
                 pbar.close()

        # Apply final gradients if there are any remaining accumulated gradients
        step += 1 # To get the total number of steps
        remaining_steps = step % GRADIENT_ACCUMULATION_STEPS
        if remaining_steps > 0:
            if jax.process_index() == 0:
                print(f"\nApplying final accumulated gradients ({remaining_steps} remaining steps)...")
            p_train_state = p_apply_grads(p_train_state)
        
        # Ensure all processes synchronize before saving
        jax.block_until_ready(p_train_state)

        # Save the final checkpoint from the main process only
        if jax.process_index() == 0:
            if num_train_steps:
                final_step = num_train_steps * NUM_EPOCHS
                ckpt_manager.save(step=final_step, items=p_train_state)
            else:
                ckpt_manager.save(step=global_step_counter, items=p_train_state)
            
            ckpt_manager.wait_until_finished()
            print("Final checkpoint saved and write-operation confirmed.")

        if jax.process_index() == 0:
            print("\nTraining complete.")

if __name__ == "__main__":
    from finetuning.pretokenize_dataset import pretokenize_and_save
    pretokenize_and_save()
    main()



# import jax
# import jax.numpy as jnp
# import optax
# import orbax.checkpoint as ocp
# from flax.training import train_state
# from flax.struct import field  # Import 'field' for static attributes
# from tqdm import tqdm
# import time
# from pathlib import Path
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

# # Custom TrainState with static fields for non-array attributes
# class TrainState(train_state.TrainState):
#     accum_grads: any
#     # Mark apply_fn and tx as static so JAX knows not to process them as arrays.
#     apply_fn: callable = field(pytree_node=False)
#     tx: optax.GradientTransformation = field(pytree_node=False)

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

# def train_step(state, batch, base_dropout_rng, step_num):
#     step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

#     def loss_fn(params):
#         logits = state.apply_fn(
#             {"params": params},
#             tokens=batch["input_ids"],
#             segment_pos=batch["segment_pos"],
#             rngs={"dropout": step_dropout_key}
#         )[0]
#         loss = calculate_loss(logits, batch["labels"])
#         return loss

#     grad_fn = jax.value_and_grad(loss_fn)
#     loss, grads = grad_fn(state.params)
#     state = state.replace(accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads))
#     return state, loss

# def apply_accumulated_gradients(state):
#     avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
#     state = state.apply_gradients(grads=avg_grads)
#     state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
#     return state


# # def main():
# #     if jax.process_index() == 0:
# #         print("JAX distributed initialized.")
# #         print(f"Total processes: {jax.process_count()}")
# #         print(f"Local devices: {jax.local_device_count()}")
# #         print(f"Global devices: {jax.device_count()}")

# #     num_devices = jax.device_count()
# #     with Mesh(jax.devices(), axis_names=('data_axis',)) as device_mesh:
# #         effective_batch_size = BATCH_SIZE
# #         if BATCH_SIZE == 1:
# #             effective_batch_size = 2
# #             if jax.process_index() == 0:
# #                 print("\n" + "="*80)
# #                 print("WARNING: Temporarily overriding per-device BATCH_SIZE to 2 to avoid a library bug.")
# #                 print("="*80 + "\n")

# #         data_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec('data_axis',))
# #         replicated_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec())

# #         def get_param_sharding(param_pytree):
# #             def get_spec(param):
# #                 if param.ndim > 1 and param.size > 1_000_000:
# #                     sharding_spec = [None] * (param.ndim - 1) + ['data_axis']
# #                     return PartitionSpec(*sharding_spec)
# #                 else:
# #                     return PartitionSpec()
# #             return jax.tree.map(get_spec, param_pytree)

# #         model, _, params, _ = load_recurrent_gemma_model(
# #             CKPT_DIR, TOK_FILE, params_dtype=WEIGHT_DTYPE
# #         )

# #         class ScanShardingHelper:
# #             def __init__(self, mesh):
# #                 self.mesh = mesh
# #                 self.sequence_axis_name = None
# #                 self.sequence_axis_index_groups = None
# #                 self.activations_sharding_spec = PartitionSpec('data_axis')
# #                 self.rnn_state_sharding_spec = PartitionSpec('data_axis')

# #         model.scan_sharding_spec = ScanShardingHelper(mesh=device_mesh)

# #         optimizer = optax.adafactor(learning_rate=LEARNING_RATE)
# #         param_sharding_rules = get_param_sharding(params)
# #         dummy_opt_state = optimizer.init(params)

# #         if isinstance(dummy_opt_state, tuple):
# #             opt_state_sharding_rules = tuple(get_param_sharding(s) for s in dummy_opt_state)
# #         else:
# #             opt_state_sharding_rules = get_param_sharding(dummy_opt_state)

# #         # Define sharding rules only for the array fields of the state.
# #         state_sharding_spec = TrainState(
# #             step=PartitionSpec(),
# #             apply_fn=None,
# #             params=param_sharding_rules,
# #             tx=None,
# #             opt_state=opt_state_sharding_rules,
# #             accum_grads=param_sharding_rules
# #         )

# #         if jax.process_index() == 0:
# #             print("Creating initial training state on CPU...")
# #         train_state_on_cpu = TrainState.create(
# #             apply_fn=model.apply,
# #             params=params,
# #             tx=optimizer,
# #             accum_grads=jax.tree.map(jnp.zeros_like, params)
# #         )
# #         del params, dummy_opt_state

# #         sharding_for_put = jax.tree.map(
# #             lambda spec: NamedSharding(device_mesh, spec),
# #             state_sharding_spec,
# #             is_leaf=lambda x: isinstance(x, PartitionSpec)
# #         )

# #         if jax.process_index() == 0:
# #             print("Sharding state across all devices...")
# #         p_train_state = jax.device_put(train_state_on_cpu, sharding_for_put)

# #         p_train_step = pjit(
# #             train_step,
# #             in_shardings=(state_sharding_spec, data_sharding, replicated_sharding, replicated_sharding),
# #             out_shardings=(state_sharding_spec, replicated_sharding)
# #         )
# #         p_apply_grads = pjit(
# #             apply_accumulated_gradients,
# #             in_shardings=(state_sharding_spec,),
# #             out_shardings=state_sharding_spec,
# #             donate_argnums=(0,)
# #         )

# #         rng = jax.random.PRNGKey(0)
# #         base_dropout_rng = jax.random.fold_in(rng, jax.process_index())
# #         ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
# #         train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)
        
# #         try:
# #             num_train_steps = len(train_dataset) // GRADIENT_ACCUMULATION_STEPS
# #         except TypeError:
# #             num_train_steps = None

# #         if jax.process_index() == 0:
# #             print("Starting training with Model Parallelism (pjit)...")

# #         for epoch in range(NUM_EPOCHS):
# #             if jax.process_index() == 0:
# #                 print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
# #                 pbar = tqdm(train_dataset, total=num_train_steps, desc=f"Epoch {epoch + 1}")
# #             else:
# #                 pbar = train_dataset

# #             total_loss = 0
# #             global_step_counter = 0
            
# #             for step, batch in enumerate(pbar):
# #                 batch = jax.tree.map(lambda x: x.numpy(), batch)
# #                 sharded_batch = jax.device_put(batch, data_sharding)

# #                 p_train_state, loss = p_train_step(p_train_state, sharded_batch, base_dropout_rng, step)
# #                 total_loss += loss.mean()

# #                 if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
# #                     p_train_state = p_apply_grads(p_train_state)
# #                     if jax.process_index() == 0:
# #                         avg_loss = total_loss.item() / GRADIENT_ACCUMULATION_STEPS
# #                         pbar.set_postfix(loss=f"{avg_loss:.4f}")
# #                         if num_train_steps is not None:
# #                             pbar.update(1)
# #                         total_loss = 0
# #                         global_step_counter += 1

# #             if jax.process_index() == 0 and num_train_steps is None:
# #                  pbar.n = global_step_counter
# #                  pbar.total = global_step_counter
# #                  pbar.close()

# #         step += 1
# #         remaining_steps = step % GRADIENT_ACCUMULATION_STEPS
# #         if remaining_steps > 0:
# #             if jax.process_index() == 0:
# #                 print(f"\nApplying final accumulated gradients ({remaining_steps} remaining steps)...")
# #             p_train_state = p_apply_grads(p_train_state)
        
# #         jax.block_until_ready(p_train_state)

# #         if jax.process_index() == 0:
# #             if num_train_steps:
# #                 final_step = num_train_steps * NUM_EPOCHS
# #                 ckpt_manager.save(step=final_step, items=p_train_state)
# #             else:
# #                 ckpt_manager.save(step=global_step_counter, items=p_train_state)
            
# #             ckpt_manager.wait_until_finished()
# #             print("Final checkpoint saved and write-operation confirmed.")

# #         if jax.process_index() == 0:
# #             print("\nTraining complete.")

# # if __name__ == "__main__":
# #     from finetuning.pretokenize_dataset import pretokenize_and_save
# #     pretokenize_and_save()
# #     main()



# def main():
#     if jax.process_index() == 0:
#         print("JAX distributed initialized.")
#         print(f"Total processes: {jax.process_count()}; Local devices: {jax.local_device_count()}; Global devices: {jax.device_count()}")

#     num_devices = jax.device_count()
#     with Mesh(jax.devices(), axis_names=('data_axis',)) as device_mesh:
#         effective_batch_size = BATCH_SIZE
#         if BATCH_SIZE == 1:
#             effective_batch_size = 2
#             if jax.process_index() == 0:
#                 print("\n" + "="*80)
#                 print("WARNING: Temporarily overriding per-device BATCH_SIZE to 2 to avoid a library bug.")
#                 print("="*80 + "\n")

#         data_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec('data_axis',))

#         def get_param_sharding(param_pytree):
#             def get_spec(param):
#                 if param.ndim > 1 and param.size > 1_000_000:
#                     sharding_spec = [None] * (param.ndim - 1) + ['data_axis']
#                     return PartitionSpec(*sharding_spec)
#                 else:
#                     return PartitionSpec()
#             return jax.tree.map(get_spec, param_pytree)

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
        
#         train_dataset_for_size = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)
#         total_train_steps = (len(train_dataset_for_size) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS

#         if jax.process_index() == 0:
#             print(f"Total training optimizer steps: {total_train_steps}")

#         lr_schedule = optax.cosine_decay_schedule(
#             init_value=LEARNING_RATE, decay_steps=total_train_steps, alpha=0.1
#         )
#         optimizer = optax.chain(
#             optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr_schedule)
#         )
        
#         if jax.process_index() == 0:
#             print("Creating initial training state on CPU...")
#         state_on_cpu = TrainState.create(
#             apply_fn=model.apply,
#             params=params,
#             tx=optimizer,
#             accum_grads=jax.tree.map(jnp.zeros_like, params)
#         )
#         del params

#         # Create sharding rule PyTrees for params and opt_state
#         param_rules = get_param_sharding(state_on_cpu.params)
#         # For opt_state, you might need to handle cases where it's a tuple of states
#         if isinstance(state_on_cpu.opt_state, tuple):
#              opt_state_rules = tuple(jax.tree.map(get_param_sharding, s) for s in state_on_cpu.opt_state)
#         else:
#              opt_state_rules = jax.tree.map(get_param_sharding, state_on_cpu.opt_state)

#         # Create the sharding specification for the entire state using the state itself as a template.
#         # This is the PyTree of PartitionSpecs that pjit needs.
#         state_sharding_spec = state_on_cpu.replace(
#             step=PartitionSpec(),
#             params=param_rules,
#             opt_state=opt_state_rules,
#             accum_grads=param_rules
#         )
        
#         # Create the concrete sharding object for device_put
#         sharding_for_put = jax.tree.map(
#             lambda spec: NamedSharding(device_mesh, spec),
#             state_sharding_spec,
#             is_leaf=lambda x: isinstance(x, PartitionSpec)
#         )

#         if jax.process_index() == 0:
#             print("Sharding state across all devices...")
#         p_train_state = jax.device_put(state_on_cpu, sharding_for_put)

#         # --- CORRECTED PJIT DEFINITIONS ---
#         p_train_step = pjit(
#             train_step,
#             in_shardings=(state_sharding_spec, data_sharding, None, None),
#             out_shardings=(state_sharding_spec, None)
#         )
#         p_apply_grads = pjit(
#             apply_accumulated_gradients,
#             in_shardings=(state_sharding_spec,),
#             out_shardings=state_sharding_spec,
#             donate_argnums=(0,)
#         )
#         # --- END OF CORRECTION ---

#         rng = jax.random.PRNGKey(0)
#         base_dropout_rng = jax.random.fold_in(rng, jax.process_index())
#         ckpt_manager = ocp.CheckpointManager(CHECKPOINT_DIR, ocp.PyTreeCheckpointer())
#         train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)
        
#         try:
#             num_train_steps_per_epoch = len(train_dataset) // GRADIENT_ACCUMULATION_STEPS
#         except TypeError:
#             num_train_steps_per_epoch = None

#         if jax.process_index() == 0:
#             print("Starting training with AdamW optimizer...")

#         for epoch in range(NUM_EPOCHS):
#             if jax.process_index() == 0:
#                 print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#                 pbar = tqdm(train_dataset, total=num_train_steps_per_epoch, desc=f"Epoch {epoch + 1}")
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
#                         current_lr = lr_schedule(p_train_state.step)
#                         pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.6f}")
#                         if num_train_steps_per_epoch is not None:
#                             pbar.update(1)
#                         total_loss = 0
#                         global_step_counter += 1

#             if jax.process_index() == 0 and num_train_steps_per_epoch is None:
#                  pbar.n = global_step_counter
#                  pbar.total = global_step_counter
#                  pbar.close()

#         step += 1
#         remaining_steps = step % GRADIENT_ACCUMULATION_STEPS
#         if remaining_steps > 0:
#             if jax.process_index() == 0:
#                 print(f"\nApplying final accumulated gradients ({remaining_steps} remaining steps)...")
#             p_train_state = p_apply_grads(p_train_state)
        
#         jax.block_until_ready(p_train_state)

#         if jax.process_index() == 0:
#             if num_train_steps_per_epoch:
#                 final_step = num_train_steps_per_epoch * NUM_EPOCHS
#                 ckpt_manager.save(step=final_step, items=p_train_state)
#             else:
#                 ckpt_manager.save(step=global_step_counter, items=p_train_state)
            
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
from flax.struct import field
from tqdm import tqdm
import time
from pathlib import Path
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.pjit import pjit
import jax.tree_util

# Assume these are in separate files as per your structure
from utils.model_loader import load_recurrent_gemma_model
from finetuning.data_pipeline import get_dataset
from finetuning.config import (
    CKPT_DIR,
    TOK_FILE,
    TRAIN_SPLIT,
    LEARNING_RATE,
    BATCH_SIZE,
    NUM_EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    CHECKPOINT_DIR,
    WEIGHT_DTYPE,
    DATASET_PROPORTION,
)

# Custom TrainState with static fields for non-array JAX-tree components
class TrainState(train_state.TrainState):
    accum_grads: any
    # Mark apply_fn and tx as static so JAX knows not to process them as arrays
    apply_fn: callable = field(pytree_node=False)
    tx: optax.GradientTransformation = field(pytree_node=False)

@jax.jit
def calculate_loss(logits, labels):
    vocab_size = logits.shape[-1]
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    loss_mask = (labels_flat != -100)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits=logits_flat.astype(jnp.float32), labels=labels_flat)
    masked_losses = jnp.where(loss_mask, losses, 0.0)
    return jnp.sum(masked_losses) / (jnp.sum(loss_mask) + 1e-8)

def train_step(state, batch, base_dropout_rng, step_num):
    step_dropout_key = jax.random.fold_in(base_dropout_rng, step_num)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            tokens=batch["input_ids"],
            segment_pos=batch["segment_pos"],
            rngs={"dropout": step_dropout_key}
        )[0]
        return calculate_loss(logits, batch["labels"])

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.replace(accum_grads=jax.tree.map(lambda x, y: x + y, state.accum_grads, grads))
    return state, loss

def apply_accumulated_gradients(state):
    avg_grads = jax.tree.map(lambda x: x / GRADIENT_ACCUMULATION_STEPS, state.accum_grads)
    state = state.apply_gradients(grads=avg_grads)
    state = state.replace(accum_grads=jax.tree.map(jnp.zeros_like, state.accum_grads))
    return state

def main():
    if jax.process_index() == 0:
        print("JAX distributed initialized.")
        print(f"Total processes: {jax.process_count()}; Global devices: {jax.device_count()}")

    num_devices = jax.device_count()
    with Mesh(jax.devices(), axis_names=('data_axis',)) as device_mesh:
        effective_batch_size = BATCH_SIZE
        if BATCH_SIZE == 1:
            effective_batch_size = 2
            if jax.process_index() == 0:
                print("\nWARNING: Temporarily overriding per-device BATCH_SIZE to 2 to avoid a library bug.\n")

        data_sharding = NamedSharding(mesh=device_mesh, spec=PartitionSpec('data_axis',))

        def get_param_sharding(param_pytree):
            def get_spec(param):
                if param.ndim > 1 and param.size > 1_000_000:
                    return PartitionSpec(None, 'data_axis')
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

        train_dataset = get_dataset(TRAIN_SPLIT, effective_batch_size * num_devices)
        steps_per_epoch = len(train_dataset) // GRADIENT_ACCUMULATION_STEPS
        total_train_steps = steps_per_epoch * NUM_EPOCHS

        if jax.process_index() == 0:
            print(f"Total training optimizer steps: {total_train_steps}")

        lr_schedule = optax.cosine_decay_schedule(
            init_value=LEARNING_RATE, decay_steps=total_train_steps, alpha=0.1
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule)
        )
        
        if jax.process_index() == 0:
            print("Creating initial training state on CPU...")
        state_on_cpu = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            accum_grads=jax.tree.map(jnp.zeros_like, params)
        )
        del params

        # Create sharding rule PyTrees for params and opt_state
        param_rules = get_param_sharding(state_on_cpu.params)
        opt_state_rules = jax.tree.map(get_param_sharding, state_on_cpu.opt_state, is_leaf=lambda x: isinstance(x, dict) and 'm' in x)

        # Create the sharding specification for the entire state.
        # This PyTree of PartitionSpecs is used to tell pjit how to handle the state.
        state_sharding_spec = state_on_cpu.replace(
            step=PartitionSpec(),
            params=param_rules,
            opt_state=opt_state_rules,
            accum_grads=param_rules
        )
        
        # Create the concrete sharding object for device_put
        sharding_for_put = jax.tree.map(
            lambda spec: NamedSharding(device_mesh, spec),
            state_sharding_spec,
            is_leaf=lambda x: isinstance(x, PartitionSpec)
        )

        if jax.process_index() == 0:
            print("Sharding state across all devices...")
        p_train_state = jax.device_put(state_on_cpu, sharding_for_put)

        # Use the state_sharding_spec object for pjit's in_shardings
        p_train_step = pjit(
            train_step,
            in_shardings=(state_sharding_spec, data_sharding, None, None),
            out_shardings=(state_sharding_spec, None)
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

        if jax.process_index() == 0:
            print("Starting training with AdamW optimizer...")

        for epoch in range(NUM_EPOCHS):
            if jax.process_index() == 0:
                print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
                pbar = tqdm(train_dataset, total=steps_per_epoch, desc=f"Epoch {epoch + 1}")
            else:
                pbar = train_dataset

            total_loss = 0
            # Convert tf.data.Dataset to an iterator
            dataset_iter = pbar if jax.process_index() == 0 else iter(train_dataset)

            for step, batch in enumerate(dataset_iter):
                # Data is already on device from tf.data, no need for jax.device_put
                p_train_state, loss = p_train_step(p_train_state, batch, base_dropout_rng, p_train_state.step)
                total_loss += loss

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    p_train_state = p_apply_grads(p_train_state)
                    if jax.process_index() == 0:
                        avg_loss = total_loss.item() / GRADIENT_ACCUMULATION_STEPS
                        current_lr = lr_schedule(p_train_state.step)
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{current_lr:.6f}")
                        if steps_per_epoch is not None:
                            pbar.update(1)
                        total_loss = 0

            if jax.process_index() == 0 and steps_per_epoch is None:
                 pbar.close()

        # Apply final gradients if necessary
        step += 1
        remaining_steps = step % GRADIENT_ACCUMULATION_STEPS
        if remaining_steps > 0:
            if jax.process_index() == 0:
                print(f"\nApplying final accumulated gradients ({remaining_steps} remaining steps)...")
            p_train_state = p_apply_grads(p_train_state)
        
        jax.block_until_ready(p_train_state)

        if jax.process_index() == 0:
            ckpt_manager.save(step=p_train_state.step, items=p_train_state)
            ckpt_manager.wait_until_finished()
            print("Final checkpoint saved and write-operation confirmed.")
            print("\nTraining complete.")

if __name__ == "__main__":
    from finetuning.pretokenize_dataset import pretokenize_and_save
    pretokenize_and_save()
    main()
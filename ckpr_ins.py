from pathlib import Path
import jax
from flax.traverse_util import flatten_dict
import orbax.checkpoint as ocp

# --- Corrected Script ---

# 1. Point to the TOP-LEVEL checkpoint manager directory
manager_path = Path("finetuning_checkpoints/").resolve()
print(f"Inspecting checkpoints in: {manager_path}")

# 2. Use CheckpointManager to find the latest step
manager = ocp.CheckpointManager(manager_path)
latest_step = manager.latest_step()

if latest_step is None:
    raise FileNotFoundError("No checkpoints found in the directory.")

print(f"Loading latest step: {latest_step}")

# 3. Define the path to the actual saved item
# This is where the _METADATA file lives
item_path = manager.directory / str(latest_step) / "default"

# 4. Use PyTreeCheckpointer to restore the saved object
# This restores the entire TrainState (as a dictionary)
restored_state = ocp.PyTreeCheckpointer().restore(item_path)

# 5. Extract the 'params' from the restored state
# The object you saved was a TrainState, which contains the params
params = restored_state['params']

# 6. List every parameter name and shape (your original logic)
print("\n--- Model Parameters ---")
for key_tuple, array in flatten_dict(params).items():
    print(".".join(key_tuple), "→", array.shape)

# 7. (Optional) Total parameter count
total = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\nTotal parameters: {total:,}")
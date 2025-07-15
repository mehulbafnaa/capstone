

from pathlib import Path

import jax
from flax.traverse_util import flatten_dict
from recurrentgemma.jax import load_parameters, GriffinConfig, Griffin

# 1. Point to your checkpoint directory (absolute)
ckpt_path = Path("2b/2b").resolve()
print(f"Loading checkpoint from: {ckpt_path}")

# 2. Load the Flax params onto one device
params = load_parameters(checkpoint_path=str(ckpt_path), sharding="single_device")

# 3. Build matching model
config = GriffinConfig.from_flax_params_or_variables(params)
model = Griffin(config)

# 4. List every parameter name and shape
for key_tuple, array in flatten_dict(params).items():
    print(".".join(key_tuple), "→", array.shape)

# 5. (Optional) Total parameter count
total = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"\nTotal parameters: {total:,}")

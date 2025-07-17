from pathlib import Path
import jax.numpy as jnp

# Base directory for the repository (assuming script is run from repo root or a subdirectory)
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Model and Tokenizer Paths (consistent with inference.py)
CKPT_DIR = REPO_ROOT / "2b" / "2b"
TOK_FILE = REPO_ROOT / "2b" / "tokenizer.model"

# Dataset Configuration
DATASET_NAME = "FrenzyMath/Herald_proofs"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
PRETOKENIZED_DATASET_DIR = REPO_ROOT / "pretokenized_dataset"

# Finetuning Hyperparameters
LEARNING_RATE = 1e-5

BATCH_SIZE = 8 # Per-device batch size. Reduced to 1 to prevent OOM errors.
NUM_EPOCHS = 1
MAX_SEQ_LEN = 512 # Maximum sequence length for tokenization

GRADIENT_ACCUMULATION_STEPS = 8
MAX_TRAIN_EXAMPLES = 8_000

# Checkpointing Configuration
CHECKPOINT_DIR = REPO_ROOT / "finetuning_checkpoints"
SAVE_EVERY_N_STEPS = 1000

# Evaluation Configuration
EVAL_EVERY_N_STEPS = 5000

# JAX/TPU Configuration
# Use bfloat16 for mixed-precision training on TPUs.
WEIGHT_DTYPE = jnp.bfloat16
ACTIVATION_DTYPE = jnp.bfloat16

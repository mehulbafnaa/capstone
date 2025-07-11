from pathlib import Path
import jax.numpy as jnp

# Base directory for the repository (assuming script is run from repo root or a subdirectory)
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Model and Tokenizer Paths (consistent with inference.py)
CKPT_DIR = REPO_ROOT / "2b-it" / "2b-it"
TOK_FILE = REPO_ROOT / "2b-it" / "tokenizer.model"

# Dataset Configuration
DATASET_NAME = "FrenzyMath/Herald_proofs"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
PRETOKENIZED_DATASET_DIR = REPO_ROOT / "pretokenized_dataset"

DATASET_PROPORTION = 0.5



# Finetuning Hyperparameters
LEARNING_RATE = 1e-5
# This is the PER-DEVICE batch size. The global batch size will be
# BATCH_SIZE * jax.device_count().
# A batch size of 4 with a sequence length of 2048 should be safe on a TPU v4.
BATCH_SIZE = 4 # Per-device batch size. Reduced to 1 to prevent OOM errors.
NUM_EPOCHS = 1
MAX_SEQ_LEN = 128 # Maximum sequence length for tokenization

GRADIENT_ACCUMULATION_STEPS = 64

# Checkpointing Configuration
CHECKPOINT_DIR = REPO_ROOT / "finetuning_checkpoints"
SAVE_EVERY_N_STEPS = 1000

# Evaluation Configuration
EVAL_EVERY_N_STEPS = 5000

# JAX/TPU Configuration
# Use bfloat16 for mixed-precision training on TPUs.
WEIGHT_DTYPE = jnp.bfloat16
ACTIVATION_DTYPE = jnp.bfloat16

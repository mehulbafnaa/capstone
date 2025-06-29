from pathlib import Path

# Base directory for the repository (assuming script is run from repo root or a subdirectory)
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Model and Tokenizer Paths (consistent with inference.py)
CKPT_DIR = REPO_ROOT / "2b-it" / "2b-it"
TOK_FILE = REPO_ROOT / "2b-it" / "tokenizer.model"

# Dataset Configuration
DATASET_NAME = "FrenzyMath/Herald_proofs"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation" # Assuming a validation split exists or will be created

# Finetuning Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 32 # Per-device batch size
NUM_EPOCHS = 3
MAX_SEQ_LEN = 2048 # Maximum sequence length for tokenization
GRADIENT_ACCUMULATION_STEPS = 1 # Set to >1 if per-device batch size is too small

# Checkpointing Configuration
CHECKPOINT_DIR = REPO_ROOT / "finetuning_checkpoints"
SAVE_EVERY_N_STEPS = 1000

# Evaluation Configuration
EVAL_EVERY_N_STEPS = 5000

# JAX/TPU Configuration
# These are typically set via environment variables, but included for clarity
# TPU_PROCESS_ID = 0 # Example
# TPU_PROCESS_COUNT = 1 # Example
# TPU_NAME = "local" # Example

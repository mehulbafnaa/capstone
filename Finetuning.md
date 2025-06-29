# Finetuning Plan for RecurrentGemma on TPU v4-8

## Objective
Develop a robust and efficient finetuning pipeline for the RecurrentGemma model. The primary goal is to adapt the model to the `FrenzyMath/Herald_proofs` dataset, specifically to improve its ability to generate correct Lean 4 theorem proofs. This pipeline will be optimized for a TPU v4-8 environment, leveraging distributed training capabilities.

## Prerequisites
*   **Python Environment:** Python 3.10+ with `uv` for dependency management.
*   **Dependencies:** All packages listed in `pyproject.toml` (especially `jax[tpu]`, `flax`, `recurrentgemma`, `datasets`, `orbax-checkpoint`, `optax`, `tqdm`).
*   **TPU Access:** A configured TPU v4-8 instance accessible via JAX.
*   **Pre-trained Assets:** The existing RecurrentGemma model checkpoints (`2b-it/2b-it`) and SentencePiece tokenizer model (`2b-it/tokenizer.model`).
*   **Dataset:** The `FrenzyMath/Herald_proofs` dataset, ideally downloaded locally using `download_dataset.py`.

## Key Considerations for TPU v4-8 Finetuning

1.  **Distributed Training (JAX `pmap`):**
    *   **Multi-Host, Multi-Device:** A TPU v4-8 consists of 4 hosts, each with 2 chips, totaling 8 devices. The finetuning script *must* leverage JAX's `pmap` (parallel map) to distribute computation and data across all 8 cores for maximum efficiency.
    *   **JAX Process Management:** Correctly initialize JAX distributed (`jax.distributed.initialize()`) and manage `jax.process_index()`, `jax.process_count()`, `jax.device_count()`, and `jax.local_device_count()`.
    *   **Parameter Replication:** Model parameters will be replicated across all devices.
    *   **Gradient Aggregation:** Gradients computed on each device must be aggregated (e.g., using `jax.lax.pmean`) before applying updates.

2.  **Memory Management:**
    *   **`bfloat16` Precision:** TPUs excel at `bfloat16` (BF16) operations. Training in BF16 will significantly reduce memory footprint and speed up computation. JAX automatically handles this for many operations when `jax_enable_x64` is false and `jax_default_matmul_precision` is set appropriately.
    *   **Gradient Accumulation:** If the desired effective batch size exceeds what fits into TPU memory, implement gradient accumulation to simulate larger batches by accumulating gradients over several micro-batches.
    *   **Batch Sizing:** Carefully determine the per-device batch size to fit within memory while maximizing utilization.

3.  **Efficient Data Loading:**
    *   **`tf.data.Dataset`:** Leverage `tensorflow.data.Dataset` for building a high-performance input pipeline. This allows for efficient shuffling, batching, and prefetching of data to keep the TPUs busy.
    *   **Tokenization on CPU:** Perform tokenization and padding on the CPU before transferring data to the TPU to avoid bottlenecking the accelerators.

4.  **Checkpointing & Resumption:**
    *   **Orbax Checkpoint:** Utilize `orbax.checkpoint` for robustly saving and restoring model parameters, optimizer state, and training progress. This is crucial for resuming interrupted training or saving intermediate models.
    *   **Periodic Saving:** Save checkpoints at regular intervals (e.g., every N steps or after each epoch).

## Plan Outline

### Phase 1: Environment Setup & Data Pipeline Development

**Goal:** Establish a stable TPU environment and create an efficient data loading and preprocessing pipeline.

1.  **Environment Verification (`main.py` / Initial Checks):**
    *   **Action:** Run `main.py` to confirm JAX correctly initializes and detects all 8 devices of the TPU v4-8.
    *   **Expected Output:** `Global devices: 8` (or 16 if it's a v3-16, but for v4-8 it should be 8).
    *   **Consideration:** Ensure `TPU_PROCESS_ID`, `TPU_PROCESS_COUNT`, and `TPU_NAME` environment variables are correctly set in the TPU execution environment.

2.  **Data Ingestion & Preprocessing (`finetuning/data_pipeline.py`):**
    *   **Action:** Create `finetuning/data_pipeline.py` to handle dataset loading, tokenization, and batching.
    *   **Steps:**
        *   **Load Dataset:** Use `datasets.load_dataset("FrenzyMath/Herald_proofs", split="train")`.
        *   **Load Tokenizer:** Instantiate `sentencepiece.SentencePieceProcessor` with `2b-it/tokenizer.model`.
        *   **Define Input/Output Format:**
            *   **Input:** `tokenizer.bos_id() + prompt_tokens + theorem_tokens + tokenizer.eos_id()`
            *   **Target Output:** `proof_tokens + tokenizer.eos_id()` (for causal language modeling).
            *   **Prompt Construction:** The prompt should guide the model to generate only the proof tactics, similar to evaluation prompts.
        *   **Tokenization Function:**
            *   Convert text to token IDs.
            *   Handle `max_seq_len` (e.g., 2048 or 4096 tokens, depending on model capacity and memory). Truncate longer sequences, pad shorter ones to `max_seq_len`.
            *   Generate an attention mask (1 for real tokens, 0 for padding).
        *   **`tf.data.Dataset` Creation:**
            *   Use `tf.data.Dataset.from_generator` or `tf.data.Dataset.from_tensor_slices` to create a dataset of tokenized examples.
            *   Apply `dataset.shuffle()`, `dataset.batch(per_device_batch_size)`, `dataset.prefetch(tf.data.AUTOTUNE)`.
            *   **Distributed Batching:** For `pmap`, ensure `dataset.batch()` creates batches that are divisible by `jax.local_device_count()` (or `jax.device_count()` for global batching).
        *   **Dataset Splitting:** Split the loaded dataset into training and validation sets (e.g., 90% train, 10% validation).

### Phase 2: Model Definition & Training Loop Implementation

**Goal:** Implement the core finetuning logic, leveraging JAX's distributed capabilities.

1.  **Model Loading for Finetuning (`finetuning/train.py` / `utils/model_loader.py`):**
    *   **Action:** Adapt the existing model loading logic to load the pre-trained RecurrentGemma parameters.
    *   **Steps:**
        *   Load `params` using `orbax.checkpoint.PyTreeCheckpointer().restore()`.
        *   Instantiate `recurrentgemma.jax.Griffin` with the appropriate `GriffinConfig` (e.g., `rg.Preset.RECURRENT_GEMMA_2B_V1`).
        *   **Crucial:** Ensure the model is set up for *training* (e.g., if there are dropout layers, they should be active). JAX models are often stateless, so this is handled by the training step function.

2.  **Finetuning Objective & Loss (`finetuning/train.py`):**
    *   **Action:** Define the loss function and metrics.
    *   **Steps:**
        *   **Task:** Causal Language Modeling (predicting the next token in the proof sequence).
        *   **Loss Function:** Use `optax.softmax_cross_entropy_with_integer_labels` for token prediction. Mask out loss contributions from padding tokens and prompt tokens.
        *   **Metrics:** Track perplexity and/or token accuracy.

3.  **Optimizer & Learning Rate Schedule (`finetuning/train.py` / `finetuning/config.py`):**
    *   **Action:** Choose and configure an optimizer and learning rate schedule.
    *   **Steps:**
        *   **Optimizer:** `optax.adamw` is a common choice for language models.
        *   **Learning Rate Schedule:** Implement a schedule with a warmup phase followed by a decay (e.g., cosine decay) using `optax.warmup_cosine_decay_schedule`.
        *   **Configuration:** Define hyperparameters (learning rate, weight decay, warmup steps) in `finetuning/config.py`.

4.  **Distributed Training Step (`finetuning/train.py`):**
    *   **Action:** Define a single training step function that operates on a batch of data and is parallelized across devices.
    *   **Steps:**
        *   **`TrainState`:** Use `optax.TrainState` to manage model parameters, optimizer state, and a PRNG key.
        *   **`@jax.jit` and `@jax.pmap`:**
            *   Define a `train_step` function that takes `TrainState` and a batch of data.
            *   Decorate `train_step` with `@jax.jit` for compilation.
            *   Wrap the `train_step` with `@jax.pmap` to execute it in parallel on all devices.
        *   **Gradient Calculation:** Use `jax.value_and_grad` to compute loss and gradients.
        *   **Gradient Aggregation:** Apply `jax.lax.pmean` to sum gradients across all devices.
        *   **Parameter Update:** Use `optax.apply_updates` to update model parameters.
        *   **PRNG Key Sharding:** Crucially, shard the PRNG key for each device using `jax.random.split` to ensure independent randomness.

5.  **Training Loop (`finetuning/train.py`):**
    *   **Action:** Implement the main training loop.
    *   **Steps:**
        *   Initialize `TrainState` and `CheckpointManager`.
        *   Iterate over epochs.
        *   Inside each epoch, iterate over the batched training dataset.
        *   Call the `pmapped_train_step` function.
        *   **Gradient Accumulation (if needed):** If using, accumulate gradients over `N` micro-batches before applying `pmean` and `apply_updates`.
        *   **Logging:** Log training loss, learning rate, and other relevant metrics to console/file/TensorBoard.
        *   **Checkpointing:** Periodically save the `TrainState` using `CheckpointManager`.

### Phase 3: Evaluation, Iteration & Deployment Considerations

**Goal:** Evaluate the finetuned model's performance and prepare for future use.

1.  **Evaluation Integration (`finetuning/train.py` / `evals/unified_eval.py`):**
    *   **Action:** Periodically evaluate the finetuned model on the validation set.
    *   **Steps:**
        *   Load the latest finetuned checkpoint.
        *   Adapt the core logic from `evals/unified_eval.py` (or directly call it if it can be configured to run on a specific validation split) to run inference and Lean verification on the validation set.
        *   Report key metrics: proof verification success rate, average inference time.

2.  **Hyperparameter Tuning:**
    *   **Action:** Iteratively adjust hyperparameters to optimize performance.
    *   **Steps:**
        *   Experiment with different learning rates, batch sizes, `max_seq_len`, number of training epochs, and prompt variations.
        *   Consider using tools like Optuna or Ray Tune for more systematic hyperparameter search if resources allow.

3.  **Deployment Considerations:**
    *   **Action:** Plan for how the finetuned model will be used for inference.
    *   **Steps:**
        *   Ensure the finetuned model can be loaded and used for inference in a non-training environment (e.g., in `inference.py` or a dedicated serving script).
        *   Consider saving the finetuned model in a format suitable for deployment (e.g., a single checkpoint file containing only the parameters).

## Proposed Directory Structure

```
capstone/
├─── finetuning/
│    ├─── __init__.py
│    ├─── train.py             # Main finetuning script (training loop, model definition)
│    ├─── data_pipeline.py     # Data loading, tokenization, and preprocessing utilities
│    └─── config.py            # Configuration for hyperparameters
├─── utils/                   # (As per previous refactoring plan - to be implemented)
│    ├─── model_loader.py      # Centralized model/tokenizer loading
│    ├─── lean_verifier.py     # Centralized Lean verification logic
│    ├─── dataset_loader.py    # Centralized dataset loading (if different from data_pipeline)
│    └─── prompt_builder.py    # Centralized prompt construction
├─── evals/
│    └─── unified_eval.py      # Primary evaluation script (after consolidation)
├─── ... (other existing files)
```

## Next Steps
1.  **Create Directory Structure:** Create the `finetuning/` directory.
2.  **Implement `finetuning/data_pipeline.py`:** Focus on efficient data loading, tokenization, and batching for TPU.
3.  **Draft `finetuning/train.py`:** Start with model loading, optimizer setup, and the basic JAX `pmap`-enabled training step.
4.  **Consult Official RecurrentGemma Repository:** Actively review the finetuning examples and best practices provided in `google-deepmind/recurrentgemma` to inform implementation details, especially regarding JAX/Flax training loops and TPU optimization.

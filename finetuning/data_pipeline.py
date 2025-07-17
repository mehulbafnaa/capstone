

import tensorflow as tf
from datasets import load_from_disk
from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN
import jax

def get_dataset(split: str, global_batch_size: int, shuffle: bool = True):
    """
    Loads the dataset and prepares a unique, deterministically
    shuffled shard for each host.
    """
    try:
        dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
    except FileNotFoundError:
        raise RuntimeError(
            f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
            "Please run `python -m finetuning.pretokenize_dataset` first."
        )

    num_examples = len(dataset)

    # Shard the Hugging Face Dataset object directly
    dataset = dataset.shard(
        num_shards=jax.process_count(),
        index=jax.process_index()
    )

    def add_masks_and_pos(example):
        example['attention_mask'] = [1] * len(example['input_ids'])
        example['segment_pos'] = list(range(len(example['input_ids'])))
        return example
    
    dataset = dataset.map(add_masks_and_pos, num_proc=1)

    # Convert the pre-sharded dataset to tf.data format
    tf_dataset = dataset.to_tf_dataset(
        columns=['input_ids', 'labels', 'attention_mask', 'segment_pos']
    )

    # --- FINAL FIX ---
    # Enforce strict determinism in the tf.data pipeline.
    # This disables any background parallel optimizations that can cause non-determinism.
    options = tf.data.Options()
    options.experimental_deterministic = True
    tf_dataset = tf_dataset.with_options(options)
    # --- END OF FIX ---

    # Now, shuffle each process's local data shard.
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=10_000, seed=42)

    # Batch and pad the data
    per_host_batch_size = global_batch_size // jax.process_count()
    tf_dataset = tf_dataset.padded_batch(
        per_host_batch_size,
        padded_shapes={
            "input_ids": [MAX_SEQ_LEN],
            "labels": [MAX_SEQ_LEN],
            "attention_mask": [MAX_SEQ_LEN],
            "segment_pos": [MAX_SEQ_LEN],
        },
        padding_values={
            "input_ids": tf.constant(0, dtype=tf.int64),
            "attention_mask": tf.constant(0, dtype=tf.int64),
            "labels": tf.constant(-100, dtype=tf.int64),
            "segment_pos": tf.constant(0, dtype=tf.int64),
        },
        drop_remainder=True
    )
    
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset, num_examples

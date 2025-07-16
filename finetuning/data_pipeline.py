# In finetuning/data_pipeline.py

import tensorflow as tf
from datasets import load_from_disk
from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN
import jax

def get_dataset(split: str, global_batch_size: int, shuffle: bool = True):
    """
    Loads the dataset and returns a correctly shuffled and sharded
    tf.data.Dataset object for multi-host training.
    """
    try:
        dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
    except FileNotFoundError:
        raise RuntimeError(
            f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
            "Please run `python -m finetuning.pretokenize_dataset` first."
        )

    num_examples = len(dataset)

    # 1. Use datasets.map() to create the new columns.
    def add_masks_and_pos(example):
        example['attention_mask'] = [1] * len(example['input_ids'])
        example['segment_pos'] = list(range(len(example['input_ids'])))
        return example

    dataset = dataset.map(add_masks_and_pos, num_proc=16)

    # 2. Convert to a tf.data.Dataset.
    tf_dataset = dataset.to_tf_dataset(
        columns=['input_ids', 'labels', 'attention_mask', 'segment_pos']
    )

    # 3. Apply the shuffle-then-shard pattern to the tf.data.Dataset.
    if shuffle:
        # Shuffle the full dataset with a fixed seed for reproducibility.
        tf_dataset = tf_dataset.shuffle(buffer_size=10_000, seed=42)

    # Shard the dataset so each host gets a unique slice.
    tf_dataset = tf_dataset.shard(
        num_shards=jax.process_count(),
        index=jax.process_index()
    )

    # Calculate the batch size for each host.
    per_host_batch_size = global_batch_size // jax.process_count()

    # Apply padding and batching.
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

    # Prefetch for performance.
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset, num_examples
# # In finetuning/data_pipeline.py

# import tensorflow as tf
# from datasets import load_from_disk
# from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN
# import jax

# def get_dataset(split: str, global_batch_size: int, shuffle: bool = True):
#     """
#     Loads the dataset and returns a correctly shuffled and sharded
#     tf.data.Dataset object for multi-host training.
#     """
#     try:
#         dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
#     except FileNotFoundError:
#         raise RuntimeError(
#             f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
#             "Please run `python -m finetuning.pretokenize_dataset` first."
#         )

#     num_examples = len(dataset)

#     # 1. Use a single-process map to deterministically add the new columns.
#     def add_masks_and_pos(example):
#         example['attention_mask'] = tf.ones_like(example['input_ids'], dtype=tf.int64)
#         example['segment_pos'] = tf.range(tf.shape(example['input_ids'])[0], dtype=tf.int64)
#         return example
        
#     # Convert to a tf.data.Dataset BEFORE mapping.
#     tf_dataset = tf.data.Dataset.from_generator(
#         lambda: iter(dataset),
#         output_signature={
#             "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int64),
#             "labels": tf.TensorSpec(shape=(None,), dtype=tf.int64),
#         }
#     ).map(add_masks_and_pos, num_parallel_calls=tf.data.AUTOTUNE)


#     # 2. Now, apply the shuffle-then-shard pattern.
#     if shuffle:
#         tf_dataset = tf_dataset.shuffle(buffer_size=10_000, seed=42)

#     tf_dataset = tf_dataset.shard(
#         num_shards=jax.process_count(),
#         index=jax.process_index()
#     )

#     per_host_batch_size = global_batch_size // jax.process_count()

#     tf_dataset = tf_dataset.padded_batch(
#         per_host_batch_size,
#         padded_shapes={
#             "input_ids": [MAX_SEQ_LEN],
#             "labels": [MAX_SEQ_LEN],
#             "attention_mask": [MAX_SEQ_LEN],
#             "segment_pos": [MAX_SEQ_LEN],
#         },
#         padding_values={
#             "input_ids": tf.constant(0, dtype=tf.int64),
#             "attention_mask": tf.constant(0, dtype=tf.int64),
#             "labels": tf.constant(-100, dtype=tf.int64),
#             "segment_pos": tf.constant(0, dtype=tf.int64),
#         },
#         drop_remainder=True
#     )
    
#     tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

#     return tf_dataset, num_examples



# In finetuning/data_pipeline.py

import jax
import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_from_disk, Dataset
from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN

def get_dataset(split: str, global_batch_size: int, shuffle: bool = True):
    """
    Loads the dataset and uses a PyTorch DistributedSampler to create a
    truly deterministic data loader for each host.
    """
    try:
        dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
    except FileNotFoundError:
        raise RuntimeError(
            f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
            "Please run `python -m finetuning.pretokenize_dataset` first."
        )

    num_examples = len(dataset)

    # 1. The DistributedSampler is the key. It tells each host exactly which
    # indices to load, removing any chance of data mismatch.
    sampler = DistributedSampler(
        dataset,
        num_replicas=jax.process_count(),
        rank=jax.process_index(),
        shuffle=shuffle,
        seed=42  # Use a seed for reproducible shuffling
    )

    # 2. Define a custom collate function to handle padding on the fly.
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Manually pad each field
        padded_input_ids = [ids + [0] * (MAX_SEQ_LEN - len(ids)) for ids in input_ids]
        padded_labels = [lbl + [-100] * (MAX_SEQ_LEN - len(lbl)) for lbl in labels]
        
        # Create attention_mask and segment_pos
        attention_mask = [[1] * len(ids) + [0] * (MAX_SEQ_LEN - len(ids)) for ids in input_ids]
        segment_pos = [list(range(len(ids))) + [0] * (MAX_SEQ_LEN - len(ids)) for ids in input_ids]

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "segment_pos": torch.tensor(segment_pos, dtype=torch.long),
        }

    # 3. Create the DataLoader with the sampler and collate_fn.
    per_host_batch_size = global_batch_size // jax.process_count()
    
    data_loader = DataLoader(
        dataset,
        batch_size=per_host_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4, # Use a few workers for performance
    )

    return data_loader, num_examples
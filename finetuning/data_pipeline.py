
# # In finetuning/data_pipeline.py

# import tensorflow as tf
# from datasets import load_from_disk
# from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN
# import jax

# def get_dataset(split: str, global_batch_size: int, shuffle: bool = True):
#     """
#     Loads the dataset and prepares a unique, deterministically
#     shuffled shard for each host.
#     """
#     try:
#         dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
#     except FileNotFoundError:
#         raise RuntimeError(
#             f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
#             "Please run `python -m finetuning.pretokenize_dataset` first."
#         )

#     num_examples = len(dataset)

#     # 1. Shuffle the Hugging Face Dataset first using a seed. This is a deterministic operation.
#     if shuffle:
#         dataset = dataset.shuffle(seed=42)

#     # 2. Map to create the missing columns. num_proc=1 to avoid fork-related non-determinism.
#     def add_masks_and_pos(example):
#         example['attention_mask'] = [1] * len(example['input_ids'])
#         example['segment_pos'] = list(range(len(example['input_ids'])))
#         return example
    
#     dataset = dataset.map(add_masks_and_pos)

#     # 3. Convert the fully prepared and shuffled dataset to tf.data format.
#     tf_dataset = dataset.to_tf_dataset(
#         columns=['input_ids', 'labels', 'attention_mask', 'segment_pos']
#     )

#     # 4. Now, shard the deterministic stream of data.
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
from datasets import load_from_disk
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

    # The DistributedSampler is the key. It gives each host a deterministic
    # list of indices to use for an epoch.
    sampler = DistributedSampler(
        dataset,
        num_replicas=jax.process_count(),
        rank=jax.process_index(),
        shuffle=shuffle,
        seed=42
    )

    # This function batches and pads the data on the fly for the DataLoader.
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

    # The per-host batch size is the global size divided by the number of hosts.
    per_host_batch_size = global_batch_size // jax.process_count()
    
    data_loader = DataLoader(
        dataset,
        batch_size=per_host_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4, # Use CPU cores to prepare data
    )

    return data_loader, num_examples
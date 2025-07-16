# import tensorflow as tf
# from datasets import load_from_disk
# from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN

# def get_dataset(split: str, batch_size: int, shuffle: bool = True):
#     """
#     Loads the pre-tokenized dataset from disk and prepares it for training.
#     """
#     try:
#         dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
#     except FileNotFoundError:
#         raise RuntimeError(
#             f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
#             "Please run `python -m finetuning.pretokenize_dataset` first."
#         )

#     def pad_and_format(examples):
#         input_ids = []
#         labels = []
#         attention_mask = []
#         segment_pos = []

#         for i in range(len(examples["input_ids"])):            # The pre-tokenization script already handled truncation and label masking.
#             # Here, we just need to pad to MAX_SEQ_LEN.
#             current_input_ids = examples["input_ids"][i]
#             current_labels = examples["labels"][i]

#             pad_len = MAX_SEQ_LEN - len(current_input_ids)
            
#             padded_input_ids = current_input_ids + [0] * pad_len
#             padded_labels = current_labels + [-100] * pad_len
#             padded_attention_mask = [1] * len(current_input_ids) + [0] * pad_len
#             # The segment_pos needs to be created here.
#             padded_segment_pos = list(range(len(current_input_ids))) + [0] * pad_len

#             input_ids.append(padded_input_ids)
#             labels.append(padded_labels)
#             attention_mask.append(padded_attention_mask)
#             segment_pos.append(padded_segment_pos)

#         return {
#             "input_ids": input_ids,
#             "labels": labels,
#             "attention_mask": attention_mask,
#             "segment_pos": segment_pos
#         }

#     # Apply padding and formatting
#     processed_dataset = dataset.map(
#         pad_and_format,
#         batched=True,
#         remove_columns=dataset.column_names,
#         load_from_cache_file=False
#     )

#     # Convert to TensorFlow dataset
#     tf_dataset = processed_dataset.to_tf_dataset(
#         columns=["input_ids", "labels", "attention_mask", "segment_pos"],
#         batch_size=batch_size,
#         drop_remainder=True, # Important for pmap
#     )

#     if shuffle:
#         tf_dataset = tf_dataset.shuffle(buffer_size=10000) # Increased buffer for better shuffling

#     tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
#     return tf_dataset

# if __name__ == "__main__":
#     from finetuning.config import TRAIN_SPLIT, BATCH_SIZE, TOK_FILE
#     import sentencepiece as spm

#     print("Testing data pipeline with pre-tokenized data...")
    
#     # Run the pre-tokenization script if data doesn't exist
#     from finetuning.pretokenize_dataset import pretokenize_and_save
#     pretokenize_and_save()

#     train_dataset = get_dataset(TRAIN_SPLIT, BATCH_SIZE)
#     vocab = spm.SentencePieceProcessor(model_file=str(TOK_FILE))

#     for batch in train_dataset.take(1):
#         print("\nSample Batch:")
#         print(f"input_ids shape: {batch['input_ids'].shape}")
#         print(f"labels shape: {batch['labels'].shape}")
#         print(f"attention_mask shape: {batch['attention_mask'].shape}")
#         print(f"segment_pos shape: {batch['segment_pos'].shape}")
        
#         # Find first non-masked label to decode for verification
#         first_real_label_idx = -1
#         for i, label in enumerate(batch["labels"][0].numpy()):
#             if label != -100:
#                 first_real_label_idx = i
#                 break
        
#         print(f"\nPrompt length (masked labels): {first_real_label_idx}")
        
#         # Decode the input up to the start of the proof
#         decoded_prompt = vocab.decode(batch["input_ids"][0].numpy().tolist()[:first_real_label_idx], skip_special_tokens=False)
#         print("\nDecoded prompt (first example):\n", decoded_prompt)

#         # Decode the part of the input that corresponds to the proof
#         decoded_proof_input = vocab.decode(batch["input_ids"][0].numpy().tolist()[first_real_label_idx:], skip_special_tokens=False)
#         print("\nDecoded proof input (first example):\n", decoded_proof_input)

#         # Decode the labels, which should match the proof
#         decoded_labels = vocab.decode([token for token in batch["labels"][0].numpy().tolist() if token != -100], skip_special_tokens=False)
#         print("\nDecoded target labels (first example, proof part):\n", decoded_labels)

#     print("Data pipeline test complete.")

import tensorflow as tf
from datasets import load_from_disk
from finetuning.config import PRETOKENIZED_DATASET_DIR, MAX_SEQ_LEN
import jax

def get_dataset(split: str, global_batch_size: int, shuffle: bool = True):
    """
    Loads the pre-tokenized dataset and prepares a unique, deterministically
    shuffled shard for each host with the correct batch size.
    """
    try:
        dataset = load_from_disk(PRETOKENIZED_DATASET_DIR)[split]
    except FileNotFoundError:
        raise RuntimeError(
            f"Pre-tokenized dataset not found at {PRETOKENIZED_DATASET_DIR}. "
            "Please run `python -m finetuning.pretokenize_dataset` first."
        )

    tf_dataset = tf.data.Dataset.from_generator(
        lambda: iter(dataset),
        output_signature={
            "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "labels": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "segment_pos": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        }
    )

    if shuffle:
        # 1. Shuffle the full dataset first with a fixed seed for reproducibility.
        tf_dataset = tf_dataset.shuffle(buffer_size=10_000, seed=42)

    # 2. Shard the identically-shuffled dataset.
    tf_dataset = tf_dataset.shard(
        num_shards=jax.process_count(),
        index=jax.process_index()
    )

    # --- THIS IS THE CRITICAL FIX ---
    # Calculate the batch size for each host.
    per_host_batch_size = global_batch_size // jax.process_count()
    # --- END OF FIX ---

    tf_dataset = tf_dataset.padded_batch(
        per_host_batch_size, # Use the per-host batch size
        padded_shapes={
            "input_ids": [MAX_SEQ_LEN],
            "attention_mask": [MAX_SEQ_LEN],
            "labels": [MAX_SEQ_LEN],
            "segment_pos": [MAX_SEQ_LEN],
        },
        padding_values={
            "input_ids": 0, "attention_mask": 0, "labels": -100, "segment_pos": 0,
        },
        drop_remainder=True
    )
    
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


from pathlib import Path
import tensorflow as tf
import sentencepiece as spm
from datasets import load_dataset
from finetuning.config import TOK_FILE, MAX_SEQ_LEN

# Load the tokenizer once
vocab = spm.SentencePieceProcessor(model_file=str(TOK_FILE))

def tokenize_function(examples):
    # Combine header, formal_theorem, and formal_proof into a single text for tokenization
    # The model will learn to generate the formal_proof given the header and formal_theorem
    # We will use a specific prompt format for finetuning

    # Input for the model: <BOS> prompt + header + formal_theorem <EOS>
    # Target for the model: <BOS> prompt + header + formal_theorem + formal_proof <EOS>
    # The loss will be computed only on the formal_proof part.

    # Construct the prompt for the model to complete the proof
    # This format should ideally match the evaluation prompt format
    prompts = []
    targets = []
    for i in range(len(examples["header"])):
        header = examples["header"][i]
        formal_theorem = examples["formal_theorem"][i]
        formal_proof = examples["formal_proof"][i]

        # Input for the model (what it sees before generating)
        prompt_input = f"Complete the following Lean 4 theorem proof:\n\n{header}\n\n{formal_theorem} := by\n  "
        
        # Full sequence for target (what the model should output, including the input part)
        full_target_text = f"Complete the following Lean 4 theorem proof:\n\n{header}\n\n{formal_theorem} := by\n  {formal_proof}"

        prompts.append(prompt_input)
        targets.append(full_target_text)

    # Tokenize prompts and targets
    # Add BOS and EOS tokens
    tokenized_prompts = vocab.encode(prompts, add_bos=True, add_eos=True)
    tokenized_targets = vocab.encode(targets, add_bos=True, add_eos=True)

    input_ids = []
    labels = []
    attention_mask = []

    for i in range(len(tokenized_prompts)):
        prompt_ids = tokenized_prompts[i]
        target_ids = tokenized_targets[i]

        # Ensure target_ids starts with prompt_ids
        if not target_ids[:len(prompt_ids)] == prompt_ids:
            # This should ideally not happen if prompt_input is a prefix of full_target_text
            # Handle cases where it might, e.g., by re-encoding or skipping
            print(f"Warning: Prompt IDs not a prefix of Target IDs for example {i}. Skipping.")
            continue

        # Truncate if necessary
        if len(target_ids) > MAX_SEQ_LEN:
            target_ids = target_ids[:MAX_SEQ_LEN]
            prompt_ids = prompt_ids[:MAX_SEQ_LEN] # Ensure prompt is also truncated consistently

        # Pad to MAX_SEQ_LEN
        padded_input_ids = prompt_ids + [0] * (MAX_SEQ_LEN - len(prompt_ids))
        padded_labels = target_ids + [0] * (MAX_SEQ_LEN - len(target_ids))
        padded_attention_mask = [1] * len(prompt_ids) + [0] * (MAX_SEQ_LEN - len(prompt_ids))

        # For causal language modeling, labels for the input part are usually ignored (-100)
        # and only the generated part contributes to the loss.
        # Here, we set labels for the prompt part to -100 so they are ignored in loss calculation.
        for j in range(len(prompt_ids)):
            if j < len(prompt_ids):
                padded_labels[j] = -100

        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        attention_mask.append(padded_attention_mask)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def get_dataset(dataset_name: str, split: str, batch_size: int, shuffle: bool = True):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    # Map the tokenization function over the dataset
    # Use batched=True to process multiple examples at once for efficiency
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False # Set to True for faster subsequent runs
    )

    # Convert to TensorFlow dataset
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=["input_ids", "labels", "attention_mask"],
        collate_fn=tf.data.DefaultAttrs(
            batch_size=batch_size,
            drop_remainder=True, # Important for pmap
            output_signature={
                "input_ids": tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32),
                "labels": tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32),
            }
        )
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000) # Adjust buffer_size as needed

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    return tf_dataset

if __name__ == "__main__":
    from finetuning.config import DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE
    print("Testing data pipeline...")
    train_dataset = get_dataset(DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE)
    
    for batch in train_dataset.take(1):
        print("\nSample Batch:")
        print(f"input_ids shape: {batch["input_ids"].shape}")
        print(f"labels shape: {batch["labels"].shape}")
        print(f"attention_mask shape: {batch["attention_mask"].shape}")
        print("\nFirst example input_ids:")
        print(batch["input_ids"][0].numpy().tolist())
        print("\nFirst example labels (masked prompt tokens are -100):")
        print(batch["labels"][0].numpy().tolist())
        print("\nFirst example attention_mask:")
        print(batch["attention_mask"][0].numpy().tolist())

        # Decode a part of the input_ids to verify
        decoded_input = vocab.decode(batch["input_ids"][0].numpy().tolist(), skip_special_tokens=False)
        print("\nDecoded input (first example):\n", decoded_input)

        # Decode the generated part (where labels are not -100)
        decoded_labels = vocab.decode([token for token in batch["labels"][0].numpy().tolist() if token != -100], skip_special_tokens=False)
        print("\nDecoded target (first example, proof part):\n", decoded_labels)

    print("Data pipeline test complete.")

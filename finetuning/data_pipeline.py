from pathlib import Path
import tensorflow as tf
import sentencepiece as spm
from datasets import load_dataset
from finetuning.config import TOK_FILE, MAX_SEQ_LEN

# Load the tokenizer once
vocab = spm.SentencePieceProcessor(model_file=str(TOK_FILE))

def tokenize_function(examples):
    # The model will learn to generate the formal_proof given the header and formal_theorem.
    # We use a standard causal language modeling setup.

    # The input to the model is the full sequence:
    # <BOS> prompt + header + formal_theorem + formal_proof <EOS>
    # The labels are the same sequence, but with the prompt part masked out.
    # The loss will be computed only on the formal_proof part.

    prompts = []
    targets = []
    for i in range(len(examples["header"])):
        header = examples["header"][i]
        formal_theorem = examples["formal_theorem"][i]
        formal_proof = examples["formal_proof"][i]

        # This is the part of the sequence that the model sees, but for which loss is not calculated.
        prompt_section = f"Complete the following Lean 4 theorem proof:\n\n{header}\n\n{formal_theorem} := by\n  "
        
        # This is the full sequence for both input and target.
        full_target_text = f"{prompt_section}{formal_proof}"

        prompts.append(prompt_section)
        targets.append(full_target_text)

    # Tokenize prompts and targets.
    # The prompt does NOT get an EOS token. The target does.
    tokenized_prompts = vocab.encode(prompts, add_bos=True, add_eos=False)
    tokenized_targets = vocab.encode(targets, add_bos=True, add_eos=True)

    input_ids = []
    labels = []
    attention_mask = []
    segment_pos = []

    for i in range(len(tokenized_prompts)):
        prompt_ids = tokenized_prompts[i]
        target_ids = tokenized_targets[i]

        # After fixing the tokenization, this check should pass.
        if not target_ids[:len(prompt_ids)] == prompt_ids:
            print(f"Warning: Prompt IDs not a prefix of Target IDs for example {i}. Skipping.")
            continue

        # Truncate the full target sequence if it's too long.
        if len(target_ids) > MAX_SEQ_LEN:
            target_ids = target_ids[:MAX_SEQ_LEN]
        
        # If the prompt is longer than the truncated target, truncate the prompt as well.
        if len(prompt_ids) > len(target_ids):
            prompt_ids = prompt_ids[:len(target_ids)]

        # The model's input is the entire tokenized target sequence.
        current_input_ids = target_ids
        # The labels are a copy, which we'll modify by masking the prompt.
        current_labels = list(target_ids)

        # Mask out the prompt tokens in the labels by setting them to -100.
        for j in range(len(prompt_ids)):
            current_labels[j] = -100

        # Pad all sequences to MAX_SEQ_LEN.
        pad_len = MAX_SEQ_LEN - len(current_input_ids)
        
        padded_input_ids = current_input_ids + [0] * pad_len
        # Pad labels with -100 so they are ignored in loss calculation.
        padded_labels = current_labels + [-100] * pad_len
        padded_attention_mask = [1] * len(current_input_ids) + [0] * pad_len
        padded_segment_pos = list(range(len(current_input_ids))) + [0] * pad_len

        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        attention_mask.append(padded_attention_mask)
        segment_pos.append(padded_segment_pos)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "segment_pos": segment_pos
    }

def get_dataset(dataset_name: str, split: str, batch_size: int, shuffle: bool = True):
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    # Convert to TensorFlow dataset
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=["input_ids", "labels", "attention_mask", "segment_pos"],
        batch_size=batch_size,
        drop_remainder=True, # Important for pmap
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    return tf_dataset

if __name__ == "__main__":
    from finetuning.config import DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE
    print("Testing data pipeline...")
    train_dataset = get_dataset(DATASET_NAME, TRAIN_SPLIT, BATCH_SIZE)
    
    for batch in train_dataset.take(1):
        print("\nSample Batch:")
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        print(f"attention_mask shape: {batch['attention_mask'].shape}")
        print(f"segment_pos shape: {batch['segment_pos'].shape}")
        
        # Find first non-masked label to decode for verification
        first_real_label_idx = -1
        for i, label in enumerate(batch["labels"][0].numpy()):
            if label != -100:
                first_real_label_idx = i
                break
        
        print(f"\nPrompt length (masked labels): {first_real_label_idx}")
        
        decoded_input = vocab.decode(batch["input_ids"][0].numpy().tolist())
        print("\nDecoded input (first example):\n", decoded_input)

        decoded_labels = vocab.decode([token for token in batch["labels"][0].numpy().tolist() if token != -100])
        print("\nDecoded target (first example, proof part):\n", decoded_labels)

    print("Data pipeline test complete.")

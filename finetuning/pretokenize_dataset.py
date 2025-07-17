

import os
from datasets import load_dataset, DatasetDict
import sentencepiece as spm
from tqdm import tqdm

from finetuning.config import DATASET_NAME, TRAIN_SPLIT, VALIDATION_SPLIT, TOK_FILE, MAX_SEQ_LEN, PRETOKENIZED_DATASET_DIR, MAX_TRAIN_EXAMPLES 

def pretokenize_and_save():
    """
    Loads the dataset, creates a train/validation split, tokenizes it, and saves it to disk.
    This is done once to avoid a bottleneck during training.
    """
    if os.path.exists(PRETOKENIZED_DATASET_DIR):
        print(f"Pre-tokenized dataset already exists at {PRETOKENIZED_DATASET_DIR}. Skipping.")
        return

    print("Loading tokenizer...")
    vocab = spm.SentencePieceProcessor(model_file=str(TOK_FILE))

    print(f"Loading dataset '{DATASET_NAME}'...")
    # Load the entire dataset, as it only has a 'train' split
    full_dataset = load_dataset(DATASET_NAME, split='train', trust_remote_code=True)
    full_dataset = full_dataset.select(range(MAX_TRAIN_EXAMPLES)

    # Create a 90/10 train/validation split
    print("Creating train/validation split...")
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"] # .train_test_split names the validation set 'test'

    def tokenize_function(examples):
        prompts = []
        targets = []
        for i in range(len(examples["header"])):
            header = examples["header"][i]
            formal_theorem = examples["formal_theorem"][i]
            formal_proof = examples["formal_proof"][i]

            prompt_section = f"Complete the following Lean 4 theorem proof:\n\n{header}\n\n{formal_theorem} := by\n  "
            full_target_text = f"{prompt_section}{formal_proof}"

            prompts.append(prompt_section)
            targets.append(full_target_text)

        tokenized_prompts = vocab.encode(prompts, add_bos=True, add_eos=False)
        tokenized_targets = vocab.encode(targets, add_bos=True, add_eos=True)

        input_ids = []
        labels = []

        for i in tqdm(range(len(tokenized_prompts)), desc="Processing examples", leave=False):
            prompt_ids = tokenized_prompts[i]
            target_ids = tokenized_targets[i]

            if not target_ids[:len(prompt_ids)] == prompt_ids:
                continue

            if len(target_ids) > MAX_SEQ_LEN:
                target_ids = target_ids[:MAX_SEQ_LEN]
            
            if len(prompt_ids) > len(target_ids):
                prompt_ids = prompt_ids[:len(target_ids)]

            current_input_ids = target_ids
            current_labels = list(target_ids)
            for j in range(len(prompt_ids)):
                current_labels[j] = -100

            input_ids.append(current_input_ids)
            labels.append(current_labels)

        return {"input_ids": input_ids, "labels": labels}

    print("Tokenizing training split...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
        num_proc=os.cpu_count() # Use all available CPU cores
    )

    print("Tokenizing validation split...")
    tokenized_validation = validation_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=validation_dataset.column_names,
        load_from_cache_file=False,
        num_proc=os.cpu_count() # Use all available CPU cores
    )

    print(f"Saving tokenized dataset to {PRETOKENIZED_DATASET_DIR}...")
    dataset_dict = DatasetDict({
        TRAIN_SPLIT: tokenized_train,
        VALIDATION_SPLIT: tokenized_validation
    })
    dataset_dict.save_to_disk(PRETOKENIZED_DATASET_DIR)
    
    print("Pre-tokenization complete.")

if __name__ == "__main__":
    pretokenize_and_save()

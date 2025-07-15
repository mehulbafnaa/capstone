#!/usr/bin/env python3
"""
High-Throughput Full Dataset Evaluation Script (Batched & Parallelized)

This script uses batch processing to dramatically speed up inference on the full
FrenzyMath/Herald_proofs dataset. It is optimized for multi-core accelerators
like TPUs by leveraging JAX's underlying parallelization capabilities.

It features:
- Tagged few-shot prompting for improved model guidance.
- Batch processing for high throughput.
- Robust CSV logging for all results.
- A clean, single-line progress bar via tqdm and warning suppression.
"""

import subprocess
import time
from pathlib import Path
import csv
from tqdm import tqdm
import warnings
import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset

# Resolve repository root from the script's location in `evals/`
REPO_ROOT = Path(__file__).parent.parent.resolve()


class HeraldInferenceTester:
    """
    Tests and verifies a RecurrentGemma model on the Herald Proofs dataset.
    """

    def __init__(self, output_log_file: Path):
        """Initialize the model, tokenizer, and logger."""
        print("Initializing RecurrentGemma model...")

        self.ckpt_dir = REPO_ROOT / "2b" / "2b"
        self.tok_file = REPO_ROOT / "2b" / "tokenizer.model"
        self.lean_project_path = REPO_ROOT / "lean_verifier"
        self.lean_src_path = self.lean_project_path / "LeanVerifier"
        self.output_log_file = output_log_file

        self._setup_logging()
        for p in [self.ckpt_dir, self.tok_file, self.lean_src_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required path not found: {p}")

        self._load_model()
        print("Model loaded successfully!")

    def _setup_logging(self):
        """Creates the CSV log file and writes the header."""
        try:
            with open(self.output_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "theorem_name",
                    "id",
                    "verified",
                    "inference_time_sec_per_example",
                    "error_type",
                    "generated_tactics",
                    "ground_truth"
                ])
            print(f"Logging results to {self.output_log_file}")
        except IOError as e:
            print(f"Error setting up log file: {e}")
            raise

    def log_result(self, data: dict):
        """Appends a single result to the CSV log file."""
        try:
            with open(self.output_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    data.get("theorem_name", ""),
                    data.get("id", ""),
                    data.get("verified", False),
                    data.get("inference_time_sec_per_example", 0.0),
                    data.get("error_type", ""),
                    data.get("generated_tactics", ""),
                    data.get("ground_truth", "")
                ])
        except IOError as e:
            print(f"Warning: Could not write to log file: {e}")

    def _load_model(self):
        """Loads the RecurrentGemma model and sampler."""
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        self.params = restored.get("params", restored)
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        self.sampler = rg.Sampler(
            model=self.model,
            vocab=self.vocab,
            params=self.params,
            deterministic_sampling=True,
            is_it_model=True
        )

    def load_full_dataset(self):
        """Loads and returns the entire Herald Proofs dataset as a list."""
        print("Loading the full FrenzyMath/Herald_proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            return list(dataset)
        except Exception as e:
            print(f"Fatal error loading dataset: {e}")
            return None

    def create_few_shot_prompt(self, target_example: dict, few_shot_examples: list) -> str:
        """Creates a structured few-shot prompt using XML-like tags."""
        prompt_parts = [
            "Complete the following Lean 4 theorem proof. Provide only the Lean tactics.",
            "Do not use 'sorry' or 'admit'. Do not provide any explanations, comments, or surrounding text."
        ]
        for ex in few_shot_examples:
            prompt_parts.append("\n<example>")
            proof_content = ex['formal_proof'].strip() if ex['formal_proof'] and ex['formal_proof'].strip() else "rfl"
            prompt_parts.append(f"{ex['header']}\n\n{ex['formal_theorem']} := by\n  {proof_content}")
            prompt_parts.append("</example>")
        prompt_parts.append("\n<problem>")
        prompt_parts.append(f"{target_example['header']}\n\n{target_example['formal_theorem']} := by")
        return "\n".join(prompt_parts)

    def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
        """Verifies a generated proof, checking for forbidden keywords first."""
        proof_block = full_code.split(':= by', 1)[-1]
        forbidden_keywords = ['sorry', 'admit', 'axiom ']
        if any(keyword in proof_block for keyword in forbidden_keywords):
            return {'verified': False, 'error': "forbidden_keyword"}
        safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
        temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"
        try:
            temp_lean_file.write_text(full_code, encoding='utf-8')
            proc = subprocess.run(['lake', 'build'], cwd=self.lean_project_path, capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                return {'verified': True, 'error': None}
            else:
                return {'verified': False, 'error': "compilation_error"}
        except subprocess.TimeoutExpired:
            return {'verified': False, 'error': 'timeout_error'}
        except Exception:
            return {'verified': False, 'error': 'verification_exception'}
        finally:
            if temp_lean_file.exists():
                temp_lean_file.unlink()

    def run_full_evaluation(self, num_few_shot: int = 2, batch_size: int = 64):
        """Runs the few-shot evaluation in batches for high throughput."""
        dataset = self.load_full_dataset()
        if not dataset: return

        if len(dataset) < num_few_shot:
            print(f"Error: Dataset size ({len(dataset)}) is smaller than num_few_shot ({num_few_shot}).")
            return

        few_shot_examples = dataset[:num_few_shot]
        test_dataset = dataset[num_few_shot:]

        print(f"Using the first {num_few_shot} example(s) as a fixed context.")
        print(f"Processing {len(test_dataset)} examples with a batch size of {batch_size}.")

        # Create batches
        batches = [test_dataset[i:i + batch_size] for i in range(0, len(test_dataset), batch_size)]

        for batch in tqdm(batches, desc=f"Batch Evaluation ({batch_size})"):
            try:
                # 1. Prepare a batch of prompts
                prompts = [self.create_few_shot_prompt(ex, few_shot_examples) for ex in batch]

                # 2. Run inference on the entire batch in one call
                start_time = time.time()
                inference_results = self.sampler(prompts, total_generation_steps=1000)
                inference_time_per_example = (time.time() - start_time) / len(batch)

                # 3. Process and verify results for each item in the batch
                for i, example in enumerate(batch):
                    generated_tactics = inference_results.text[i]
                    if '</problem>' in generated_tactics:
                        generated_tactics = generated_tactics.split('</problem>')[0]

                    full_generated_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_tactics}"
                    verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])

                    self.log_result({
                        "theorem_name": example['name'],
                        "id": example['id'],
                        "verified": verification_result["verified"],
                        "inference_time_sec_per_example": round(inference_time_per_example, 2),
                        "error_type": verification_result["error"],
                        "generated_tactics": generated_tactics,
                        "ground_truth": example['formal_proof']
                    })
            except Exception as e:
                print(f"\nCritical error processing a batch: {e}. Logging failures for batch.")
                for example in batch:
                    self.log_result({"theorem_name": example['name'], "id": example['id'], "verified": False, "error_type": "batch_processing_error"})


def main():
    """Main execution function to configure and run the experiment."""
    # Suppress the noisy JAX warning to keep the tqdm bar clean

    jax.distributed.initialize()

    if jax.process_index() == 0:
        print("=" * 80)
        print("JAX DISTRIBUTED SYSTEM INITIALIZED SUCESSFULLY")
        print(f"   Total processes:      {jax.process_count()}")
        print(f"   Current process ID:   {jax.process_index()}")
        print(f"   Total devices found:  {jax.device_count()}")
        print(f"   Local devices on this host: {jax.local_device_count()}")
        print("=" * 80)
    warnings.filterwarnings("ignore", message="Some donated buffers were not usable:")

    # --- Configuration ---
    # Number of examples to place in the prompt's context.
    NUM_FEW_SHOT = 2
    # Number of prompts to process in parallel. Should be a multiple of 8 for a TPU v3-8.
    # 64 is a strong default. Decrease to 32 if you encounter memory issues.
    BATCH_SIZE = 64
    
    output_filename = f"batched_{NUM_FEW_SHOT}-shot_v3-8_results.csv"
    output_file = REPO_ROOT / output_filename

    try:
        tester = HeraldInferenceTester(output_log_file=output_file)
        tester.run_full_evaluation(num_few_shot=NUM_FEW_SHOT, batch_size=BATCH_SIZE)
        print("\n" + "=" * 80)
        print("Full dataset evaluation completed!")
        print(f"Results have been logged to: {output_file}")
        print("=" * 80)
    except Exception as e:
        print(f"\nA fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    # Ensure tqdm is installed: `uv pip install tqdm`
    exit(main())
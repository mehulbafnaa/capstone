# #!/usr/bin/env python3
# """
# Full Dataset Evaluation Script (with Tagged Few-Shot Prompting)

# This script runs a configurable few-shot evaluation across the entire
# FrenzyMath/Herald_proofs dataset. It uses a structured prompt with XML-like
# tags to guide the model. It logs detailed results for each example to a
# CSV file and displays a progress bar.
# """

# import subprocess
# import time
# from pathlib import Path
# import csv
# from tqdm import tqdm  # For the progress bar

# import jax
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset

# # Path setup remains the same, looking for directories from the repo root.
# REPO_ROOT = Path(__file__).parent.parent.resolve()


# class HeraldInferenceTester:
#     """
#     Tests and verifies a RecurrentGemma model on the Herald Proofs dataset.
#     """

#     def __init__(self, output_log_file: Path):
#         """Initialize the model, tokenizer, and logger."""
#         print("Initializing RecurrentGemma model...")

#         self.ckpt_dir = REPO_ROOT / "2b-it" / "2b-it"
#         self.tok_file = REPO_ROOT / "2b-it" / "tokenizer.model"
#         self.lean_project_path = REPO_ROOT / "lean_verifier"
#         self.lean_src_path = self.lean_project_path / "LeanVerifier"

#         # Setup logging
#         self.output_log_file = output_log_file
#         self._setup_logging()

#         for p in [self.ckpt_dir, self.tok_file, self.lean_src_path]:
#             if not p.exists():
#                 raise FileNotFoundError(f"Required path not found: {p}")

#         self._load_model()
#         print("Model loaded successfully!")

#     def _setup_logging(self):
#         """Creates the CSV log file and writes the header."""
#         try:
#             with open(self.output_log_file, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     "theorem_name",
#                     "id",
#                     "verified",
#                     "inference_time_sec",
#                     "error_type",
#                     "generated_tactics",
#                     "ground_truth"
#                 ])
#             print(f"Logging results to {self.output_log_file}")
#         except IOError as e:
#             print(f"Error setting up log file: {e}")
#             raise

#     def log_result(self, data: dict):
#         """Appends a single result to the CSV log file."""
#         try:
#             with open(self.output_log_file, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     data.get("theorem_name", ""),
#                     data.get("id", ""),
#                     data.get("verified", False),
#                     data.get("inference_time_sec", 0.0),
#                     data.get("error_type", ""),
#                     data.get("generated_tactics", ""),
#                     data.get("ground_truth", "")
#                 ])
#         except IOError as e:
#             print(f"Warning: Could not write to log file: {e}")

#     def _load_model(self):
#         # This method is unchanged
#         restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
#         self.params = restored.get("params", restored)
#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
#         self.sampler = rg.Sampler(
#             model=self.model,
#             vocab=self.vocab,
#             params=self.params,
#             deterministic_sampling=True,
#             is_it_model=True
#         )

#     def load_full_dataset(self):
#         """Loads and returns the entire Herald Proofs dataset."""
#         print("Loading the full FrenzyMath/Herald_proofs dataset...")
#         try:
#             dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#             # Convert to list for easier manipulation
#             return list(dataset)
#         except Exception as e:
#             print(f"Fatal error loading dataset: {e}")
#             return None

#     def create_few_shot_prompt(self, target_example: dict, few_shot_examples: list) -> str:
#         """
#         Creates a structured few-shot prompt using tags to clearly separate
#         examples from the target problem.
#         """
#         prompt_parts = [
#             "Complete the following Lean 4 theorem proof. Provide only the Lean tactics.",
#             "Do not use 'sorry' or 'admit'. Do not provide any explanations, comments, or surrounding text."
#         ]

#         # Add the high-quality examples, wrapped in <example> tags.
#         for ex in few_shot_examples:
#             prompt_parts.append("\n<example>")
#             proof_content = ex['formal_proof'].strip() if ex['formal_proof'] and ex['formal_proof'].strip() else "rfl"
#             prompt_parts.append(f"{ex['header']}\n\n{ex['formal_theorem']} := by\n  {proof_content}")
#             prompt_parts.append("</example>")

#         # Add the final target problem, wrapped in <problem> tags.
#         prompt_parts.append("\n<problem>")
#         prompt_parts.append(f"{target_example['header']}\n\n{target_example['formal_theorem']} := by")
        
#         return "\n".join(prompt_parts)


#     def run_inference(self, prompt: str, max_steps: int = 1000) -> dict:
#         # This method is unchanged
#         start_time = time.time()
#         try:
#             result = self.sampler([prompt], total_generation_steps=max_steps)
#             inference_time = time.time() - start_time
#             return {'success': True, 'generated_tactics': result.text[0], 'inference_time': inference_time}
#         except Exception as e:
#             return {'success': False, 'error': str(e), 'inference_time': time.time() - start_time}

#     def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
#         # This method is unchanged
#         proof_block = full_code.split(':= by', 1)[-1]
#         forbidden_keywords = ['sorry', 'admit', 'axiom ']

#         if any(keyword in proof_block for keyword in forbidden_keywords):
#             return {'verified': False, 'error': "forbidden_keyword"}

#         safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
#         temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

#         try:
#             temp_lean_file.write_text(full_code, encoding='utf-8')
#             proc = subprocess.run(['lake', 'build'], cwd=self.lean_project_path, capture_output=True, text=True, timeout=120)

#             if proc.returncode == 0:
#                 return {'verified': True, 'error': None}
#             else:
#                 return {'verified': False, 'error': "compilation_error"}
#         except subprocess.TimeoutExpired:
#             return {'verified': False, 'error': 'timeout_error'}
#         except Exception:
#             return {'verified': False, 'error': 'verification_exception'}
#         finally:
#             if temp_lean_file.exists():
#                 temp_lean_file.unlink()

#     def run_full_evaluation(self, num_few_shot: int = 2):
#         """Runs the few-shot evaluation over the entire dataset."""
#         dataset = self.load_full_dataset()
#         if not dataset:
#             return

#         # Select a fixed set of examples for the few-shot context
#         if len(dataset) < num_few_shot:
#             print(f"Error: Dataset size ({len(dataset)}) is smaller than num_few_shot ({num_few_shot}).")
#             return
            
#         # We'll use the first `num_few_shot` examples from the dataset as our fixed context
#         few_shot_examples = dataset[:num_few_shot]
#         test_dataset = dataset[num_few_shot:]

#         print(f"Using the first {num_few_shot} example(s) as a fixed context for all tests.")
        
#         # Use tqdm for a progress bar
#         for example in tqdm(test_dataset, desc=f"{num_few_shot}-Shot Evaluation"):
#             log_data = {
#                 "theorem_name": example['name'],
#                 "id": example['id'],
#                 "ground_truth": example['formal_proof']
#             }

#             try:
#                 prompt = self.create_few_shot_prompt(example, few_shot_examples)
#                 inference_result = self.run_inference(prompt)
                
#                 log_data["inference_time_sec"] = round(inference_result.get('inference_time', 0.0), 2)

#                 if not inference_result['success']:
#                     log_data["verified"] = False
#                     log_data["error_type"] = "inference_error"
#                 else:
#                     generated_tactics = inference_result['generated_tactics']
#                     # Clean up potential model artifacts like closing tags
#                     if '</problem>' in generated_tactics:
#                         generated_tactics = generated_tactics.split('</problem>')[0]
                    
#                     log_data["generated_tactics"] = generated_tactics
#                     full_generated_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_tactics}"
                    
#                     verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])
#                     log_data["verified"] = verification_result["verified"]
#                     log_data["error_type"] = verification_result["error"]
                
#                 self.log_result(log_data)

#             except Exception as e:
#                 print(f"\nCritical error on example {example['name']}: {e}. Logging as critical_error.")
#                 log_data["verified"] = False
#                 log_data["error_type"] = "critical_error"
#                 self.log_result(log_data)


# def main():
#     """Main execution function."""
#     # Configure the experiment here
#     NUM_FEW_SHOT = 2  # Set the number of examples to use in the prompt (e.g., 1, 2, or 3)
#     output_filename = f"tagged_{NUM_FEW_SHOT}-shot_results.csv"
#     output_file = REPO_ROOT / output_filename
    
#     try:
#         tester = HeraldInferenceTester(output_log_file=output_file)
#         tester.run_full_evaluation(num_few_shot=NUM_FEW_SHOT)
#         print("\n" + "=" * 80)
#         print("Full dataset evaluation completed!")
#         print(f"Results have been logged to: {output_file}")
#         print("=" * 80)
#     except Exception as e:
#         print(f"\nA fatal error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
#     return 0


# if __name__ == "__main__":
#     # Before running, ensure you have tqdm installed:
#     # uv pip install tqdm
#     exit(main())


#!/usr/bin/env python3
"""
Full Dataset Evaluation Script (with Tagged Few-Shot Prompting)

This script runs a configurable few-shot evaluation across the entire
FrenzyMath/Herald_proofs dataset. It logs results to a CSV file and displays
a clean, single-line progress bar by suppressing noisy warnings.
"""

import subprocess
import time
from pathlib import Path
import csv
from tqdm import tqdm
import warnings  # <<< NEW: Import the warnings library

import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset

# Path setup remains the same
REPO_ROOT = Path(__file__).parent.parent.resolve()


class HeraldInferenceTester:
    # --- NO CHANGES ARE NEEDED IN THIS CLASS ---
    # All methods (init, load, prompt, inference, verify) are correct.
    def __init__(self, output_log_file: Path):
        print("Initializing RecurrentGemma model...")
        self.ckpt_dir = REPO_ROOT / "2b-it" / "2b-it"
        self.tok_file = REPO_ROOT / "2b-it" / "tokenizer.model"
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
        try:
            with open(self.output_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["theorem_name", "id", "verified", "inference_time_sec", "error_type", "generated_tactics", "ground_truth"])
            print(f"Logging results to {self.output_log_file}")
        except IOError as e:
            print(f"Error setting up log file: {e}")
            raise

    def log_result(self, data: dict):
        try:
            with open(self.output_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([data.get("theorem_name", ""), data.get("id", ""), data.get("verified", False), data.get("inference_time_sec", 0.0), data.get("error_type", ""), data.get("generated_tactics", ""), data.get("ground_truth", "")])
        except IOError as e:
            print(f"Warning: Could not write to log file: {e}")

    def _load_model(self):
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        self.params = restored.get("params", restored)
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=self.params, deterministic_sampling=True, is_it_model=True)

    def load_full_dataset(self):
        print("Loading the full FrenzyMath/Herald_proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            return list(dataset)
        except Exception as e:
            print(f"Fatal error loading dataset: {e}")
            return None

    def create_few_shot_prompt(self, target_example: dict, few_shot_examples: list) -> str:
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

    def run_inference(self, prompt: str, max_steps: int = 1000) -> dict:
        start_time = time.time()
        try:
            result = self.sampler([prompt], total_generation_steps=max_steps)
            inference_time = time.time() - start_time
            return {'success': True, 'generated_tactics': result.text[0], 'inference_time': inference_time}
        except Exception as e:
            return {'success': False, 'error': str(e), 'inference_time': time.time() - start_time}

    def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
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

    def run_full_evaluation(self, num_few_shot: int = 2):
        dataset = self.load_full_dataset()
        if not dataset:
            return
        if len(dataset) < num_few_shot:
            print(f"Error: Dataset size ({len(dataset)}) is smaller than num_few_shot ({num_few_shot}).")
            return
        few_shot_examples = dataset[:num_few_shot]
        test_dataset = dataset[num_few_shot:]
        print(f"Using the first {num_few_shot} example(s) as a fixed context for all tests.")
        for example in tqdm(test_dataset, desc=f"{num_few_shot}-Shot Evaluation"):
            log_data = {"theorem_name": example['name'], "id": example['id'], "ground_truth": example['formal_proof']}
            try:
                prompt = self.create_few_shot_prompt(example, few_shot_examples)
                inference_result = self.run_inference(prompt)
                log_data["inference_time_sec"] = round(inference_result.get('inference_time', 0.0), 2)
                if not inference_result['success']:
                    log_data["verified"] = False
                    log_data["error_type"] = "inference_error"
                else:
                    generated_tactics = inference_result['generated_tactics']
                    if '</problem>' in generated_tactics:
                        generated_tactics = generated_tactics.split('</problem>')[0]
                    log_data["generated_tactics"] = generated_tactics
                    full_generated_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_tactics}"
                    verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])
                    log_data["verified"] = verification_result["verified"]
                    log_data["error_type"] = verification_result["error"]
                self.log_result(log_data)
            except Exception as e:
                print(f"\nCritical error on example {example['name']}: {e}. Logging as critical_error.")
                log_data["verified"] = False
                log_data["error_type"] = "critical_error"
                self.log_result(log_data)

def main():
    """Main execution function."""
    
    # <<< NEW: Add this line to suppress the disruptive JAX warning >>>
    warnings.filterwarnings("ignore", message="Some donated buffers were not usable:")
    
    # Configure the experiment here
    NUM_FEW_SHOT = 2
    output_filename = f"tagged_{NUM_FEW_SHOT}-shot_results.csv"
    output_file = REPO_ROOT / output_filename
    
    try:
        tester = HeraldInferenceTester(output_log_file=output_file)
        tester.run_full_evaluation(num_few_shot=NUM_FEW_SHOT)
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
    exit(main())
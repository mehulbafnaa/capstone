
# #!/usr/bin/env python3
# """
# RecurrentGemma Inference on Herald Proofs Dataset

# This script runs a comprehensive test suite for a RecurrentGemma model,
# generating and verifying Lean 4 proofs from the Herald Proofs dataset.
# It is optimized for multi-host TPU environments, correctly handling
# distributed JAX arrays.
# """

# import argparse
# import json
# import logging
# import subprocess
# import tempfile
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import jax
# import jax.numpy as jnp
# import jax.sharding as jsh
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset
# from jax.experimental.multihost_utils import process_allgather

# # --- Setup Professional Logging ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# # Silence verbose JAX/HF logging on non-main processes
# if jax.process_index() != 0:
#     logging.getLogger("jax").setLevel(logging.WARNING)
#     logging.getLogger("datasets").setLevel(logging.WARNING)
#     logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# # --- Type Aliases for Clarity ---
# JaxParams = Any
# Example = Dict[str, Any]


# class HeraldInferenceTester:
#     """
#     Manages model loading, inference, and verification for Herald Proofs.

#     Attributes:
#         verifier_path (Path): Path to the Lean verifier executable directory.
#         model (rg.Griffin): The loaded RecurrentGemma model.
#         vocab (spm.SentencePieceProcessor): The tokenizer.
#         params (JaxParams): Model parameters sharded across devices.
#         sampler (rg.Sampler): The generation sampler.
#         mesh (jsh.Mesh): The JAX device mesh.
#         num_devices (int): Total number of devices in the mesh.
#     """

#     def __init__(self, ckpt_dir: Path, tok_path: Path, verifier_path: Path):
#         """Initializes the model, tokenizer, and JAX device mesh."""
#         self.log("Initializing RecurrentGemma model... 🚀")

#         if not ckpt_dir.exists():
#             raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
#         if not tok_path.exists():
#             raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")
#         if not verifier_path.exists():
#             raise FileNotFoundError(f"Lean verifier 'lake' not found in: {verifier_path}")

#         self.verifier_path = verifier_path
#         self.devices = jax.devices()
#         self.num_devices = len(self.devices)
#         self.log(f"JAX device mesh created with {self.num_devices} devices.")
#         self.mesh = jsh.Mesh(self.devices, ('data',))

#         self._load_model_and_prepare(ckpt_dir, tok_path)
#         self.log("Model loaded successfully!")

#     def _load_model_and_prepare(self, ckpt_dir: Path, tok_path: Path):
#         """Loads model weights, tokenizer, and shards parameters."""
#         restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
#         params = restored.get("params", restored)
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(
#             params, preset=rg.Preset.RECURRENT_GEMMA_2B_V1
#         )
#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(tok_path))
#         self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=params)

#         with self.mesh:
#             replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
#             self.params = jax.device_put(params, replicated_sharding)

#     def run_inference(self, prompts: List[str], max_steps: int) -> Dict[str, Any]:
#         """Runs batched inference and gathers results from all workers."""
#         self.log(f"Running inference on a batch of {len(prompts)} prompts...")
#         start_time = time.time()

#         try:
#             self.sampler.total_generation_steps = max_steps
#             tokenized_prompts = [self.vocab.encode(p, add_bos=True) for p in prompts]
#             max_prompt_len = max(len(t) for t in tokenized_prompts)
#             total_len = max_prompt_len + max_steps
#             input_tokens = jnp.array([
#                 t + [self.vocab.pad_id()] * (total_len - len(t))
#                 for t in tokenized_prompts
#             ])

#             sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data'))
#             sharded_tokens = jax.device_put(input_tokens, sharding)
#             rng = jax.random.PRNGKey(0)

#             # Generate distributed output tokens
#             output_tokens = self.sampler.sample_fn(self.params, rng, sharded_tokens)

#             # *** CRITICAL FIX for Multi-Host ***
#             # Gather the distributed array onto all hosts before using it.
#             gathered_tokens = process_allgather(output_tokens)

#             # Now, safely decode the complete, local array.
#             generated_texts = self.vocab.decode(gathered_tokens.tolist())
#             inference_time = time.time() - start_time
#             return {'success': True, 'generated_texts': generated_texts, 'inference_time': inference_time}

#         except Exception as e:
#             self.log(f"Inference failed: {e}", level="error")
#             if jax.process_index() == 0:
#                 logging.exception("Detailed inference traceback:")
#             return {'success': False, 'error': str(e), 'inference_time': time.time() - start_time}

#     def verify_lean_proof(self, full_lean_code: str) -> Tuple[bool, str]:
#         """Verifies a Lean proof using an external verifier."""
#         try:
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=True, encoding='utf-8') as temp_file:
#                 temp_file.write(full_lean_code)
#                 temp_file.flush()
#                 command = ['lake', 'exe', 'Main', temp_file.name]
#                 process = subprocess.run(
#                     command,
#                     cwd=self.verifier_path,
#                     capture_output=True,
#                     text=True,
#                     encoding='utf-8',
#                     timeout=60
#                 )
#                 is_verified = process.returncode == 0
#                 output = (process.stdout + process.stderr).strip()
#                 return is_verified, output
#         except subprocess.TimeoutExpired:
#             return False, "Error: Verification timed out after 60 seconds."
#         except Exception as e:
#             self.log(f"An unexpected error occurred during verification: {e}", level="error")
#             return False, f"An unexpected error occurred during verification: {e}"

#     def run_test_suite(self, num_examples: int, max_generation_steps: int):
#         """Executes the full test suite from loading data to reporting results."""
#         self.log("Starting Herald Proofs inference test suite...")
#         self.log("=" * 80)

#         original_examples = self._load_herald_examples(num_examples)
#         if not original_examples:
#             self.log("No examples loaded. Exiting test suite.", level="error")
#             return

#         # Pad batch to be divisible by the number of devices
#         num_to_pad = (self.num_devices - len(original_examples) % self.num_devices) % self.num_devices
#         padded_examples = original_examples + [original_examples[-1]] * num_to_pad
#         if num_to_pad > 0:
#             self.log(f"Padding batch with {num_to_pad} examples to match device count of {self.num_devices}.")

#         prompts = [self._create_prompt(ex) for ex in padded_examples]
#         inference_result = self.run_inference(prompts, max_generation_steps)
#         results = []

#         if inference_result['success']:
#             self.log(f"Batch inference completed in {inference_result['inference_time']:.2f}s")
#             generated_texts = inference_result['generated_texts'][:len(original_examples)]

#             for i, (example, text) in enumerate(zip(original_examples, generated_texts)):
#                 self.log(f"\n===== PROCESSING EXAMPLE {i+1}/{len(original_examples)}: {example['name']} =====")
#                 generated_proof = self._extract_proof_from_output(text)
#                 evaluation = self._evaluate_example(generated_proof, example['formal_proof'])

#                 self.log("🤖 Generated Proof:\n" + "-" * 40 + f"\n{generated_proof}\n" + "-" * 40)

#                 full_lean_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_proof}"
#                 is_verified, verifier_log = self.verify_lean_proof(full_lean_code)
#                 evaluation['verified'] = is_verified

#                 status_emoji = "✅" if is_verified else "❌"
#                 self.log(f"Verification Status: {status_emoji} {'Verified' if is_verified else 'Failed'}")
#                 if not is_verified:
#                     self.log(f"Verifier Output:\n{verifier_log}")

#                 results.append({
#                     'example': example,
#                     'generated_proof': generated_proof,
#                     'evaluation': evaluation,
#                     'total_batch_time': inference_result['inference_time']
#                 })
#         else:
#             self.log(f"Batch inference failed: {inference_result['error']}", level="error")
#             for example in original_examples:
#                 results.append({'example': example, 'error': inference_result['error']})

#         if jax.process_index() == 0:
#             self._print_summary(results)
#             self._save_results(results)

#     def _load_herald_examples(self, num_examples: int) -> List[Example]:
#         """Loads a diverse set of examples from the Herald Proofs dataset."""
#         self.log(f"Loading {num_examples} examples from Herald Proofs dataset...")
#         try:
#             dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#             df = dataset.to_pandas()
#             df['formal_proof_len'] = df['formal_proof'].str.len()

#             # Select a diverse range of proof lengths
#             short_proofs = df[df['formal_proof_len'] < 100]
#             medium_proofs = df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)]
#             long_proofs = df[df['formal_proof_len'] >= 300]

#             selected_indices = []
#             if not short_proofs.empty: selected_indices.append(short_proofs.index[0])
#             if not medium_proofs.empty: selected_indices.append(medium_proofs.index[0])
#             if not long_proofs.empty: selected_indices.append(long_proofs.index[0])
            
#             # Ensure we have enough examples
#             if len(selected_indices) < num_examples:
#                 remaining_indices = list(set(df.index) - set(selected_indices))
#                 selected_indices.extend(remaining_indices[:num_examples - len(selected_indices)])

#             examples = [df.iloc[idx].to_dict() for idx in selected_indices[:num_examples]]
#             for i, ex in enumerate(examples):
#                 self.log(f"  Example {i+1}: '{ex['name']}' (proof length: {len(ex['formal_proof'])} chars)")
#             return examples

#         except Exception as e:
#             self.log(f"Fatal error loading dataset: {e}", level="error")
#             return []

#     @staticmethod
#     def _create_prompt(example: Example) -> str:
#         """Creates a standardized prompt for the model."""
#         return (
#             "Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.\n\n"
#             f"{example['header']}\n\n{example['formal_theorem']} := by\n  sorry"
#         )

#     @staticmethod
#     def _extract_proof_from_output(output_text: str) -> str:
#         """Extracts the proof tactics from the model's raw output."""
#         try:
#             if ':= by' in output_text:
#                 after_by = output_text.split(':= by', 1)[1]
#                 # Filter out empty lines and the original 'sorry'
#                 proof_lines = [line.strip() for line in after_by.split('\n') if line.strip() and 'sorry' not in line]
#                 if proof_lines:
#                     return '\n  '.join(proof_lines)
#         except IndexError:
#             pass
#         return "No valid proof generated"

#     @staticmethod
#     def _evaluate_example(generated_proof: str, ground_truth: str) -> Dict[str, Any]:
#         """Compares the generated proof to the ground truth."""
#         generated_clean = "".join(generated_proof.split())
#         ground_truth_clean = "".join(ground_truth.split())
#         exact_match = generated_clean == ground_truth_clean

#         # A simple tactic similarity metric
#         common_tactics = {'rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction', 'funext', 'ext'}
#         ground_truth_tactics = {t for t in common_tactics if t in ground_truth.lower()}
#         generated_tactics = {t for t in common_tactics if t in generated_proof.lower()}
        
#         tactic_overlap = len(ground_truth_tactics.intersection(generated_tactics))
#         tactic_total = len(ground_truth_tactics) or 1
        
#         return {'exact_match': exact_match, 'tactic_similarity': tactic_overlap / tactic_total}

#     def _print_summary(self, results: List[Dict]):
#         """Prints a final summary of the test suite run."""
#         if jax.process_index() != 0: return

#         print("\n" + "=" * 80)
#         print("TEST SUITE SUMMARY")
#         print("=" * 80)

#         successful = [r for r in results if 'evaluation' in r]
#         failed = [r for r in results if 'error' in r]
        
#         print(f"Total Examples Tested: {len(results)}")
#         print(f"✅ Successful Inferences: {len(successful)}")
#         print(f"❌ Failed Inferences: {len(failed)}")

#         if successful:
#             verified_count = sum(1 for r in successful if r['evaluation'].get('verified'))
#             exact_matches = sum(1 for r in successful if r['evaluation']['exact_match'])
#             avg_sim = sum(r['evaluation']['tactic_similarity'] for r in successful) / len(successful)
            
#             print(f"\nVerified Proofs: {verified_count} / {len(successful)}")
#             print(f"Exact Text Matches: {exact_matches} / {len(successful)}")
#             print(f"Average Tactic Similarity: {avg_sim:.2%}")
#             print(f"Total Batch Inference Time: {successful[0]['total_batch_time']:.2f}s")

#         print("\n--- Individual Results ---")
#         for i, result in enumerate(results, 1):
#             name = result['example']['name']
#             if 'error' in result:
#                 status = "FAILED INFERENCE"
#             else:
#                 eval_ = result['evaluation']
#                 verified_status = "✅ VERIFIED" if eval_.get('verified') else "❌ UNVERIFIED"
#                 status = f"{verified_status} (Similarity: {eval_['tactic_similarity']:.2%})"
#             print(f"  {i}. {name:<70} {status}")
#         print("=" * 80)

#     def _save_results(self, results: List[Dict]):
#         """Saves detailed results to a JSON file if the user agrees."""
#         if jax.process_index() != 0: return

#         try:
#             choice = input("\nSave detailed results to file? (y/n): ").lower()
#             if choice.startswith('y'):
#                 output_path = Path("herald_inference_results.json")
#                 json_results = []
#                 for r in results:
#                     res = {
#                         'example_name': r['example']['name'],
#                         'example_id': r['example'].get('id'),
#                         'success': 'error' not in r
#                     }
#                     if res['success']:
#                         res.update({
#                             'generated_proof': r['generated_proof'],
#                             'verified': r['evaluation'].get('verified', False),
#                             'exact_match': r['evaluation']['exact_match'],
#                             'tactic_similarity': r['evaluation']['tactic_similarity'],
#                         })
#                     else:
#                         res['error'] = r.get('error', 'Unknown error')
#                     json_results.append(res)
                
#                 with output_path.open('w', encoding='utf-8') as f:
#                     json.dump(json_results, f, indent=2)
#                 print(f"Results saved to '{output_path.resolve()}'")
#         except (IOError, OSError) as e:
#             print(f"Error: Could not save results file. {e}")
#         except Exception:
#             # Handle cases where input() fails in a non-interactive script
#             print("\nSkipping save results in non-interactive mode.")

#     def log(self, message: str, level: str = "info"):
#         """Logs a message only on the main process."""
#         if jax.process_index() == 0:
#             if level == "info":
#                 logging.info(message)
#             elif level == "error":
#                 logging.error(message)
#             elif level == "warning":
#                 logging.warning(message)


# # def main():
# #     """Main execution function with argument parsing."""
# #     parser = argparse.ArgumentParser(description="Run RecurrentGemma inference on Herald Proofs.")
# #     parser.add_argument("--ckpt_dir", type=Path, default=Path("2b/2b"), help="Path to the model checkpoint directory.")
# #     parser.add_argument("--tok_path", type=Path, default=Path("2b/tokenizer.model"), help="Path to the tokenizer.model file.")
# #     parser.add_argument("--verifier_path", type=Path, default=Path.home()/ "capstone/lean_verifier", help="Path to the Lean verifier project directory.")
# #     parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to test from the dataset.")
# #     parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of tokens to generate.")
# #     args = parser.parse_args()

# #     try:
# #         tester = HeraldInferenceTester(
# #             ckpt_dir=args.ckpt_dir,
# #             tok_path=args.tok_path,
# #             verifier_path=args.verifier_path,
# #         )
# #         tester.run_test_suite(
# #             num_examples=args.num_examples,
# #             max_generation_steps=args.max_steps
# #         )
# #     except FileNotFoundError as e:
# #         logging.error(e)
# #         return 1
# #     except Exception as e:
# #         if jax.process_index() == 0:
# #             logging.exception(f"A fatal error occurred in the main execution: {e}")
# #         return 1
    
# #     if jax.process_index() == 0:
# #         logging.info("Test suite completed!")
# #     return 0


# def main():
#     """Main execution function with argument parsing."""
#     parser = argparse.ArgumentParser(description="Run RecurrentGemma inference on Herald Proofs.")
#     parser.add_argument("--ckpt_dir", type=Path, default=Path("2b/2b"), help="Path to the model checkpoint directory.")
#     parser.add_argument("--tok_path", type=Path, default=Path("2b/tokenizer.model"), help="Path to the tokenizer.model file.")
#     parser.add_argument("--verifier_path", type=Path, default=Path.home() / "capstone/lean_verifier", help="Path to the Lean verifier project directory.")
#     parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to test from the dataset.")
#     parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of tokens to generate.")
#     args = parser.parse_args()

#     # --- FIX: Resolve paths to be absolute ---
#     ckpt_path_abs = args.ckpt_dir.resolve()
#     tok_path_abs = args.tok_path.resolve()
#     verifier_path_abs = args.verifier_path.resolve()

#     try:
#         tester = HeraldInferenceTester(
#             ckpt_dir=ckpt_path_abs,
#             tok_path=tok_path_abs,
#             verifier_path=verifier_path_abs,
#         )
#         tester.run_test_suite(
#             num_examples=args.num_examples,
#             max_generation_steps=args.max_steps
#         )
#     except FileNotFoundError as e:
#         # The __init__ method now handles this, but this is good practice
#         if jax.process_index() == 0:
#             logging.error(e)
#         return 1
#     except Exception as e:
#         if jax.process_index() == 0:
#             logging.exception(f"A fatal error occurred in the main execution: {e}")
#         return 1
    
#     if jax.process_index() == 0:
#         logging.info("Test suite completed!")
#     return 0

# if __name__ == "__main__":
#     exit(main())




#!/usr/bin/env python3
"""
RecurrentGemma Inference on Herald Proofs Dataset

This script runs a comprehensive test suite for a RecurrentGemma model,
generating and verifying Lean 4 proofs from the Herald Proofs dataset.
It is optimized for multi-host TPU environments, correctly handling
distributed JAX arrays.
"""

import argparse
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import jax.sharding as jsh
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset
from jax.experimental.multihost_utils import process_allgather

# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Silence verbose JAX/HF logging on non-main processes
if jax.process_index() != 0:
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# --- Type Aliases for Clarity ---
JaxParams = Any
Example = Dict[str, Any]


class HeraldInferenceTester:
    """
    Manages model loading, inference, and verification for Herald Proofs.

    Attributes:
        verifier_path (Path): Path to the Lean verifier executable directory.
        model (rg.Griffin): The loaded RecurrentGemma model.
        vocab (spm.SentencePieceProcessor): The tokenizer.
        params (JaxParams): Model parameters sharded across devices.
        sampler (rg.Sampler): The generation sampler.
        mesh (jsh.Mesh): The JAX device mesh.
        num_devices (int): Total number of devices in the mesh.
    """

    def __init__(self, ckpt_dir: Path, tok_path: Path, verifier_path: Path):
        """Initializes the model, tokenizer, and JAX device mesh."""
        self.log("Initializing RecurrentGemma model... 🚀")

        if not ckpt_dir.is_absolute():
            raise ValueError(f"Checkpoint path must be absolute. Got: {ckpt_dir}")
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")
        if not verifier_path.exists():
            raise FileNotFoundError(f"Lean verifier project not found at: {verifier_path}")

        self.verifier_path = verifier_path
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        self.log(f"JAX device mesh created with {self.num_devices} devices.")
        self.mesh = jsh.Mesh(self.devices, ('data',))

        self._load_model_and_prepare(ckpt_dir, tok_path)
        self.log("Model loaded successfully!")

    def _load_model_and_prepare(self, ckpt_dir: Path, tok_path: Path):
        """Loads model weights, tokenizer, and shards parameters."""
        restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
        params = restored.get("params", restored)
        cfg = rg.GriffinConfig.from_flax_params_or_variables(
            params, preset=rg.Preset.RECURRENT_GEMMA_2B_V1
        )
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(tok_path))
        self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=params)

        with self.mesh:
            replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
            self.params = jax.device_put(params, replicated_sharding)

    def run_inference(self, prompts: List[str], max_steps: int) -> Dict[str, Any]:
        """Runs batched inference and gathers results from all workers."""
        self.log(f"Running inference on a batch of {len(prompts)} prompts...")
        start_time = time.time()

        try:
            self.sampler.total_generation_steps = max_steps
            tokenized_prompts = [self.vocab.encode(p, add_bos=True) for p in prompts]
            max_prompt_len = max(len(t) for t in tokenized_prompts)
            total_len = max_prompt_len + max_steps
            input_tokens = jnp.array([
                t + [self.vocab.pad_id()] * (total_len - len(t))
                for t in tokenized_prompts
            ])

            sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data'))
            sharded_tokens = jax.device_put(input_tokens, sharding)
            rng = jax.random.PRNGKey(0)

            # Generate distributed output tokens
            output_tokens = self.sampler.sample_fn(self.params, rng, sharded_tokens)

            # *** CRITICAL FIX for Multi-Host ***
            gathered_tokens = process_allgather(output_tokens)

            # Now, safely decode the complete, local array.
            generated_texts = self.vocab.decode(gathered_tokens.tolist())
            inference_time = time.time() - start_time
            return {'success': True, 'generated_texts': generated_texts, 'inference_time': inference_time}

        except Exception as e:
            self.log(f"Inference failed: {e}", level="error")
            if jax.process_index() == 0:
                logging.exception("Detailed inference traceback:")
            return {'success': False, 'error': str(e), 'inference_time': time.time() - start_time}

    def verify_lean_proof(self, full_lean_code: str) -> Tuple[bool, str]:
        """Verifies a Lean proof using an external verifier."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=True, encoding='utf-8') as temp_file:
                temp_file.write(full_lean_code)
                temp_file.flush()
                command = ['lake', 'exe', 'Main', temp_file.name]
                process = subprocess.run(
                    command,
                    cwd=self.verifier_path,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=60
                )
                is_verified = process.returncode == 0
                output = (process.stdout + process.stderr).strip()
                return is_verified, output
        except subprocess.TimeoutExpired:
            return False, "Error: Verification timed out after 60 seconds."
        except Exception as e:
            self.log(f"An unexpected error occurred during verification: {e}", level="error")
            return False, f"An unexpected error occurred during verification: {e}"

    def run_test_suite(self, num_examples: int, max_generation_steps: int):
        """Executes the full test suite from loading data to reporting results."""
        self.log("Starting Herald Proofs inference test suite...")
        self.log("=" * 80)

        original_examples = self._load_herald_examples(num_examples)
        if not original_examples:
            self.log("No examples loaded. Exiting test suite.", level="error")
            return

        # Pad batch to be divisible by the number of devices
        num_to_pad = (self.num_devices - len(original_examples) % self.num_devices) % self.num_devices
        padded_examples = original_examples + [original_examples[-1]] * num_to_pad
        if num_to_pad > 0:
            self.log(f"Padding batch with {num_to_pad} examples to match device count of {self.num_devices}.")

        prompts = [self._create_prompt(ex) for ex in padded_examples]
        inference_result = self.run_inference(prompts, max_generation_steps)
        results = []

        if inference_result['success']:
            self.log(f"Batch inference completed in {inference_result['inference_time']:.2f}s")
            generated_texts = inference_result['generated_texts'][:len(original_examples)]

            for i, (example, text) in enumerate(zip(original_examples, generated_texts)):
                self.log(f"\n===== PROCESSING EXAMPLE {i+1}/{len(original_examples)}: {example['name']} =====")
                generated_proof = self._extract_proof_from_output(text)
                evaluation = self._evaluate_example(generated_proof, example['formal_proof'])

                self.log("🤖 Generated Proof:\n" + "-" * 40 + f"\n{generated_proof}\n" + "-" * 40)

                full_lean_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_proof}"
                is_verified, verifier_log = self.verify_lean_proof(full_lean_code)
                evaluation['verified'] = is_verified

                status_emoji = "✅" if is_verified else "❌"
                self.log(f"Verification Status: {status_emoji} {'Verified' if is_verified else 'Failed'}")
                if not is_verified:
                    self.log(f"Verifier Output:\n{verifier_log}")

                results.append({
                    'example': example,
                    'generated_proof': generated_proof,
                    'evaluation': evaluation,
                    'total_batch_time': inference_result['inference_time']
                })
        else:
            self.log(f"Batch inference failed: {inference_result['error']}", level="error")
            for example in original_examples:
                results.append({'example': example, 'error': inference_result['error']})

        if jax.process_index() == 0:
            self._print_summary(results)
            self._save_results(results)

    def _load_herald_examples(self, num_examples: int) -> List[Example]:
        """Loads a diverse set of examples from the Herald Proofs dataset."""
        self.log(f"Loading {num_examples} examples from Herald Proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas()
            df['formal_proof_len'] = df['formal_proof'].str.len()

            short_proofs = df[df['formal_proof_len'] < 100]
            medium_proofs = df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)]
            long_proofs = df[df['formal_proof_len'] >= 300]

            selected_indices = []
            if not short_proofs.empty: selected_indices.append(short_proofs.index[0])
            if not medium_proofs.empty: selected_indices.append(medium_proofs.index[0])
            if not long_proofs.empty: selected_indices.append(long_proofs.index[0])
            
            if len(selected_indices) < num_examples:
                remaining_indices = list(set(df.index) - set(selected_indices))
                selected_indices.extend(remaining_indices[:num_examples - len(selected_indices)])

            examples = [df.iloc[idx].to_dict() for idx in selected_indices[:num_examples]]
            for i, ex in enumerate(examples):
                self.log(f"  Example {i+1}: '{ex['name']}' (proof length: {len(ex['formal_proof'])} chars)")
            return examples

        except Exception as e:
            self.log(f"Fatal error loading dataset: {e}", level="error")
            return []

    @staticmethod
    def _create_prompt(example: Example) -> str:
        """Creates a standardized prompt for the model."""
        return (
            "Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.\n\n"
            f"{example['header']}\n\n{example['formal_theorem']} := by\n  sorry"
        )

    @staticmethod
    def _extract_proof_from_output(output_text: str) -> str:
        """Extracts the proof tactics from the model's raw output."""
        try:
            if ':= by' in output_text:
                after_by = output_text.split(':= by', 1)[1]
                proof_lines = [line.strip() for line in after_by.split('\n') if line.strip() and 'sorry' not in line]
                if proof_lines:
                    return '\n  '.join(proof_lines)
        except IndexError:
            pass
        return "No valid proof generated"

    @staticmethod
    def _evaluate_example(generated_proof: str, ground_truth: str) -> Dict[str, Any]:
        """Compares the generated proof to the ground truth."""
        generated_clean = "".join(generated_proof.split())
        ground_truth_clean = "".join(ground_truth.split())
        exact_match = generated_clean == ground_truth_clean

        common_tactics = {'rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction', 'funext', 'ext'}
        ground_truth_tactics = {t for t in common_tactics if t in ground_truth.lower()}
        generated_tactics = {t for t in common_tactics if t in generated_proof.lower()}
        
        tactic_overlap = len(ground_truth_tactics.intersection(generated_tactics))
        tactic_total = len(ground_truth_tactics) or 1
        
        return {'exact_match': exact_match, 'tactic_similarity': tactic_overlap / tactic_total}

    def _print_summary(self, results: List[Dict]):
        """Prints a final summary of the test suite run."""
        if jax.process_index() != 0: return

        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)

        successful = [r for r in results if 'evaluation' in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"Total Examples Tested: {len(results)}")
        print(f"✅ Successful Inferences: {len(successful)}")
        print(f"❌ Failed Inferences: {len(failed)}")

        if successful:
            verified_count = sum(1 for r in successful if r['evaluation'].get('verified'))
            exact_matches = sum(1 for r in successful if r['evaluation']['exact_match'])
            avg_sim = sum(r['evaluation']['tactic_similarity'] for r in successful) / len(successful) if successful else 0
            
            print(f"\nVerified Proofs: {verified_count} / {len(successful)}")
            print(f"Exact Text Matches: {exact_matches} / {len(successful)}")
            print(f"Average Tactic Similarity: {avg_sim:.2%}")
            print(f"Total Batch Inference Time: {successful[0]['total_batch_time']:.2f}s")

        print("\n--- Individual Results ---")
        for i, result in enumerate(results, 1):
            name = result['example']['name']
            if 'error' in result:
                status = "FAILED INFERENCE"
            else:
                eval_ = result['evaluation']
                verified_status = "✅ VERIFIED" if eval_.get('verified') else "❌ UNVERIFIED"
                status = f"{verified_status} (Similarity: {eval_['tactic_similarity']:.2%})"
            print(f"  {i}. {name:<70} {status}")
        print("=" * 80)

    def _save_results(self, results: List[Dict]):
        """Saves detailed results to a JSON file if the user agrees."""
        if jax.process_index() != 0: return

        try:
            choice = input("\nSave detailed results to file? (y/n): ").lower()
            if choice.startswith('y'):
                output_path = Path("herald_inference_results.json")
                json_results = []
                for r in results:
                    res = {
                        'example_name': r['example']['name'],
                        'example_id': r['example'].get('id'),
                        'success': 'error' not in r
                    }
                    if res['success']:
                        res.update({
                            'generated_proof': r['generated_proof'],
                            'verified': r['evaluation'].get('verified', False),
                            'exact_match': r['evaluation']['exact_match'],
                            'tactic_similarity': r['evaluation']['tactic_similarity'],
                        })
                    else:
                        res['error'] = r.get('error', 'Unknown error')
                    json_results.append(res)
                
                with output_path.open('w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=2)
                print(f"Results saved to '{output_path.resolve()}'")
        except (IOError, OSError) as e:
            print(f"Error: Could not save results file. {e}")
        except Exception:
            print("\nSkipping save results in non-interactive mode.")

    def log(self, message: str, level: str = "info"):
        """Logs a message only on the main process."""
        if jax.process_index() == 0:
            if level == "info":
                logging.info(message)
            elif level == "error":
                logging.error(message)
            elif level == "warning":
                logging.warning(message)


def main():
    """Main execution function with argument parsing."""
    parser = argparse.ArgumentParser(description="Run RecurrentGemma inference on Herald Proofs.")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("2b/2b"), help="Path to the model checkpoint directory.")
    parser.add_argument("--tok_path", type=Path, default=Path("2b/tokenizer.model"), help="Path to the tokenizer.model file.")
    parser.add_argument("--verifier_path", type=Path, default=Path.home() / "capstone/lean_verifier", help="Path to the Lean verifier project directory.")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to test from the dataset.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of tokens to generate.")
    args = parser.parse_args()

    # --- FIX: Resolve paths to be absolute ---
    ckpt_path_abs = args.ckpt_dir.resolve()
    tok_path_abs = args.tok_path.resolve()
    verifier_path_abs = args.verifier_path.resolve()

    try:
        tester = HeraldInferenceTester(
            ckpt_dir=ckpt_path_abs,
            tok_path=tok_path_abs,
            verifier_path=verifier_path_abs,
        )
        tester.run_test_suite(
            num_examples=args.num_examples,
            max_generation_steps=args.max_steps
        )
    except (FileNotFoundError, ValueError) as e:
        if jax.process_index() == 0:
            logging.error(e)
        return 1
    except Exception as e:
        if jax.process_index() == 0:
            logging.exception(f"A fatal error occurred in the main execution: {e}")
        return 1
    
    if jax.process_index() == 0:
        logging.info("Test suite completed!")
    return 0


if __name__ == "__main__":
    exit(main())
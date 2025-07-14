


# #!/usr/bin/env python3
# """
# Herald Proofs Model Inference Test (TPU v4-16 Optimized)

# This script loads examples from the Herald Proofs dataset and runs batched
# inference using the RecurrentGemma model. It pads the batch to be divisible
# by the number of devices for efficient parallel execution.
# """

# import json
# import time
# from pathlib import Path

# import jax
# import jax.numpy as jnp
# import jax.sharding as jsh
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset


# class HeraldInferenceTester:
#     """
#     Test RecurrentGemma model on Herald Proofs dataset examples
#     """

#     def __init__(self, ckpt_dir: str = "2b/2b", tok_file: str = "2b/tokenizer.model"):
#         """Initialize the model, tokenizer, and JAX device mesh."""
#         print("Initializing RecurrentGemma model... 🚀")

#         self.ckpt_dir = Path(ckpt_dir).resolve()
#         self.tok_file = Path(tok_file).resolve()

#         self.devices = jax.devices()
#         self.num_devices = len(self.devices)
#         print(f"JAX device mesh created with {self.num_devices} devices.")

#         self.mesh = jsh.Mesh(self.devices, ('data',))

#         self._load_model_and_prepare_jit()
#         print("Model loaded and JIT-compiled successfully!")

#     def _load_model_and_prepare_jit(self):
#         """Load the RecurrentGemma model, shard parameters, and JIT-compile the generation function."""
#         restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
#         params = restored.get("params", restored)

#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

#         self.sampler = rg.Sampler(
#             model=self.model,
#             vocab=self.vocab,
#             params=params,
#         )

#         with self.mesh:
#             replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
#             self.params = jax.device_put(params, replicated_sharding)

#             self._jitted_sample_fn = jax.jit(
#                 self.sampler.sample_fn,
#                 in_shardings=(
#                     replicated_sharding,
#                     None,
#                     jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data')),
#                 ),
#                 out_shardings=jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data'))
#             )

#     def load_herald_examples(self, num_examples: int = 3):
#         """Load examples from Herald Proofs dataset."""
#         print(f"Loading {num_examples} examples from Herald Proofs dataset...")
#         try:
#             dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#             df = dataset.to_pandas()
#             df['formal_proof_len'] = df['formal_proof'].str.len()

#             short_idx = df.index[df['formal_proof_len'] < 100][0] if not df[df['formal_proof_len'] < 100].empty else 0
#             medium_idx = df.index[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)][0] if not df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)].empty else 1
#             long_idx = df.index[df['formal_proof_len'] >= 300][0] if not df[df['formal_proof_len'] >= 300].empty else 2

#             selected_indices = [short_idx, medium_idx, long_idx][:num_examples]
#             examples = [df.iloc[idx].to_dict() for idx in selected_indices]
#             for i, ex in enumerate(examples):
#                 print(f"  Example {i+1}: '{ex['name']}' (proof length: {len(ex['formal_proof'])} chars)")
#             return examples
#         except Exception as e:
#             print(f"Error loading dataset: {e}")
#             return []

#     def create_prompt(self, example):
#         """Create a prompt for the model based on a Herald dataset example."""
#         return f"Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.\n\n{example['header']}\n\n{example['formal_theorem']} := by\n  sorry"

#     def run_inference(self, prompts: list[str], max_steps: int = 1000):
#         """Run batched inference on a list of prompts."""
#         print(f"Running inference on a batch of {len(prompts)} prompts...")
#         start_time = time.time()

#         try:
#             self.sampler.total_generation_steps = max_steps
#             tokenized_prompts = [self.vocab.encode(p, add_bos=True) for p in prompts]
#             max_len = max(len(t) for t in tokenized_prompts)
#             padded_tokens = jnp.array([
#                 t + [self.vocab.pad_id()] * (max_len - len(t)) for t in tokenized_prompts
#             ])

#             rng = jax.random.PRNGKey(0)
#             output_tokens = self._jitted_sample_fn(self.params, rng, padded_tokens)
#             generated_texts = self.vocab.decode(output_tokens.tolist())
#             inference_time = time.time() - start_time

#             return {'success': True, 'generated_texts': generated_texts, 'inference_time': inference_time}
#         except Exception as e:
#             return {'success': False, 'error': str(e), 'inference_time': time.time() - start_time}

#     def extract_proof_from_output(self, output_text: str):
#         """Extract the proof portion from the model's generated output."""
#         if ':= by' in output_text:
#             try:
#                 after_by = output_text.split(':= by', 1)[1]
#                 proof_lines = [
#                     line.strip() for line in after_by.split('\n') if line.strip() and 'sorry' not in line
#                 ]
#                 if proof_lines:
#                     return '\n  '.join(proof_lines)
#             except IndexError:
#                 pass
#         return "No valid proof generated"

#     def evaluate_example(self, generated_proof, ground_truth):
#         """Perform a simple evaluation of the generated proof against the ground truth."""
#         generated_clean = generated_proof.replace(' ', '').replace('\n', '').lower()
#         ground_truth_clean = ground_truth.replace(' ', '').replace('\n', '').lower()
#         exact_match = generated_clean == ground_truth_clean

#         tactics = ['rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction']
#         ground_truth_tactics = {t for t in tactics if t in ground_truth.lower()}
#         generated_tactics = {t for t in tactics if t in generated_proof.lower()}

#         tactic_overlap = len(ground_truth_tactics.intersection(generated_tactics))
#         tactic_total = len(ground_truth_tactics) if len(ground_truth_tactics) > 0 else 1

#         return {'exact_match': exact_match, 'tactic_similarity': tactic_overlap / tactic_total}

#     def run_test_suite(self):
#         """Run the complete test suite on a batch of Herald examples."""
#         print("Starting Herald Proofs inference test suite...")
#         print("=" * 80)

#         original_examples = self.load_herald_examples(3)
#         if not original_examples:
#             print("No examples loaded. Exiting.")
#             return

#         # --- FIX: Pad the batch to be divisible by the number of devices ---
#         num_to_pad = (self.num_devices - len(original_examples) % self.num_devices) % self.num_devices
#         padded_examples = original_examples + [original_examples[-1]] * num_to_pad
#         print(f"Padding batch with {num_to_pad} examples to match device count of {self.num_devices}.")
        
#         prompts = [self.create_prompt(ex) for ex in padded_examples]
#         inference_result = self.run_inference(prompts)
        
#         results = []
#         if inference_result['success']:
#             print(f"\nBatch inference completed in {inference_result['inference_time']:.2f}s")
#             # --- FIX: Slice the results to remove the padding ---
#             generated_texts = inference_result['generated_texts'][:len(original_examples)]

#             for i, example in enumerate(original_examples):
#                 print(f"\nPROCESSING EXAMPLE {i+1}/{len(original_examples)}: {example['name']}")
#                 print("-" * 60)

#                 generated_text = generated_texts[i]
#                 generated_proof = self.extract_proof_from_output(generated_text)
#                 evaluation = self.evaluate_example(generated_proof, example['formal_proof'])

#                 print(f"Exact match: {evaluation['exact_match']}")
#                 print(f"Tactic similarity: {evaluation['tactic_similarity']:.2f}")

#                 results.append({
#                     'example': example, 'generated_text': generated_text,
#                     'generated_proof': generated_proof, 'evaluation': evaluation,
#                     'total_batch_time': inference_result['inference_time']
#                 })
#         else:
#             print(f"Batch inference failed: {inference_result['error']}")
#             for example in original_examples:
#                 results.append({'example': example, 'error': inference_result['error']})

#         self._print_summary(results)
#         return results

#     def _print_summary(self, results):
#         """Print a summary of all test results."""
#         print("\n" + "=" * 80)
#         print("TEST SUITE SUMMARY")
#         print("=" * 80)

#         successful_runs = [r for r in results if 'generated_proof' in r]
#         failed_runs = [r for r in results if 'error' in r]
#         print(f"Successful inferences: {len(successful_runs)}/{len(results)}")
#         print(f"Failed inferences: {len(failed_runs)}/{len(results)}")

#         if successful_runs:
#             total_time = successful_runs[0]['total_batch_time']
#             exact_matches = sum(1 for r in successful_runs if r['evaluation']['exact_match'])
#             avg_tactic_sim = sum(r['evaluation']['tactic_similarity'] for r in successful_runs) / len(successful_runs)
#             print(f"Total batch inference time: {total_time:.2f}s")
#             print(f"Exact matches: {exact_matches}/{len(successful_runs)}")
#             print(f"Average tactic similarity: {avg_tactic_sim:.2f}")

#         print("\nIndividual Results:")
#         for i, result in enumerate(results, 1):
#             name = result['example']['name']
#             status = "FAILED"
#             if 'evaluation' in result:
#                 match = "EXACT" if result['evaluation']['exact_match'] else "PARTIAL"
#                 sim = result['evaluation']['tactic_similarity']
#                 status = f"{match} (similarity: {sim:.2f})"
#             print(f"  {i}. {name}: {status}")
#         print("=" * 80)

# def main():
#     """Main execution function."""
#     try:
#         tester = HeraldInferenceTester()
#         results = tester.run_test_suite()

#         print("\nSave detailed results to file? (y/n): ", end="")
#         if input().lower().startswith('y'):
#             json_results = []
#             for r in results:
#                 # --- FIX: Convert NumPy types to native Python types for JSON serialization ---
#                 res = {
#                     'example_name': str(r['example']['name']),
#                     'example_id': int(r['example']['id'])
#                 }
#                 if 'error' in r:
#                     res['error'] = r.get('error', 'Unknown error')
#                 else:
#                     res.update({
#                         'generated_proof': str(r['generated_proof']),
#                         'exact_match': bool(r['evaluation']['exact_match']),
#                         'tactic_similarity': float(r['evaluation']['tactic_similarity'])
#                     })
#                 json_results.append(res)

#             with open('herald_inference_results_corrected.json', 'w') as f:
#                 json.dump(json_results, f, indent=2)
#             print("Results saved to 'herald_inference_results_corrected.json'")

#         print("\nTest suite completed!")
#         return 0
#     except Exception as e:
#         print(f"\nFatal error in main execution: {e}")
#         return 1

# if __name__ == "__main__":
#     exit(main())





#!/usr/bin/env python3
"""
Herald Proofs Model Inference Test (TPU v4-16 Optimized)

This script loads examples, generates proofs, and automatically verifies them
using an external Lean verifier tool. It pads the batch to be divisible
by the number of devices for efficient parallel execution.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.sharding as jsh
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset


class HeraldInferenceTester:
    """
    Test RecurrentGemma model on Herald Proofs dataset examples
    """

    def __init__(self, ckpt_dir: str = "2b/2b", tok_file: str = "2b/tokenizer.model"):
        """Initialize the model, tokenizer, and JAX device mesh."""
        print("Initializing RecurrentGemma model... 🚀")

        # --- Path to your lean_verifier project ---
        script_dir = Path(__file__).resolve().parent
        self.verifier_path = script_dir / 'lean_verifier'

        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.tok_file = Path(tok_file).resolve()

        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        print(f"JAX device mesh created with {self.num_devices} devices.")

        self.mesh = jsh.Mesh(self.devices, ('data',))

        self._load_model_and_prepare_jit()
        print("Model loaded and JIT-compiled successfully!")

    def _load_model_and_prepare_jit(self):
        # ... (this method remains the same as before)
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        params = restored.get("params", restored)
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=params)
        with self.mesh:
            replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
            self.params = jax.device_put(params, replicated_sharding)
            self._jitted_sample_fn = jax.jit(
                self.sampler.sample_fn,
                in_shardings=(replicated_sharding, None, jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data'))),
                out_shardings=jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data'))
            )
    
    # --- NEW METHOD TO CALL LEAN VERIFIER ---
    def verify_lean_proof(self, full_lean_code: str) -> (bool, str):
        """
        Verifies a full Lean code snippet using the external lean_verifier tool.

        Returns:
            A tuple (is_verified, output_log).
        """
        try:
            # Create a temporary file to hold the lean code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=True, encoding='utf-8') as temp_file:
                temp_file.write(full_lean_code)
                temp_file.flush() # Ensure the content is written to disk

                # Construct the command to run the verifier
                command = ['lake', 'exe', 'Main', temp_file.name]
                
                # Run the command from within the verifier's directory
                process = subprocess.run(
                    command,
                    cwd=self.verifier_path,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=60 # Add a timeout to prevent hangs
                )

                # Check the return code: 0 means success
                is_verified = process.returncode == 0
                return is_verified, process.stdout + process.stderr

        except FileNotFoundError:
            return False, f"Error: 'lake' command not found. Is Lean 4 installed and in your PATH?"
        except subprocess.TimeoutExpired:
            return False, "Error: Verification timed out."
        except Exception as e:
            return False, f"An unexpected error occurred during verification: {e}"


    def run_test_suite(self):
        """Run the complete test suite on a batch of Herald examples."""
        print("Starting Herald Proofs inference test suite...")
        print("=" * 80)

        original_examples = self.load_herald_examples(3)
        if not original_examples:
            print("No examples loaded. Exiting.")
            return

        num_to_pad = (self.num_devices - len(original_examples) % self.num_devices) % self.num_devices
        padded_examples = original_examples + [original_examples[-1]] * num_to_pad
        if num_to_pad > 0:
            print(f"Padding batch with {num_to_pad} examples to match device count of {self.num_devices}.")
        
        prompts = [self.create_prompt(ex) for ex in padded_examples]
        inference_result = self.run_inference(prompts)
        
        results = []
        if inference_result['success']:
            print(f"\nBatch inference completed in {inference_result['inference_time']:.2f}s")
            generated_texts = inference_result['generated_texts'][:len(original_examples)]

            for i, example in enumerate(original_examples):
                print(f"\n===== PROCESSING EXAMPLE {i+1}/{len(original_examples)}: {example['name']} =====")
                print("-" * 60)

                generated_text = generated_texts[i]
                generated_proof = self.extract_proof_from_output(generated_text)
                evaluation = self.evaluate_example(generated_proof, example['formal_proof'])
                
                # --- ADDED: Print generated and ground truth proofs ---
                print("🤖 Generated Proof:")
                print("-" * 40)
                print(generated_proof)
                print("-" * 40)
                
                # --- ADDED: Verification Step ---
                full_lean_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_proof}"
                is_verified, verifier_log = self.verify_lean_proof(full_lean_code)
                evaluation['verified'] = is_verified # Add to our results

                verification_status = "✅ Verified Successfully" if is_verified else "❌ Verification Failed"
                print(f"\nVerification Status: {verification_status}")
                if not is_verified:
                    print("Verifier Output:\n", verifier_log)
                print("-" * 60)

                results.append({
                    'example': example, 'generated_text': generated_text,
                    'generated_proof': generated_proof, 'evaluation': evaluation,
                    'total_batch_time': inference_result['inference_time']
                })
        else:
            print(f"Batch inference failed: {inference_result['error']}")
            for example in original_examples:
                results.append({'example': example, 'error': inference_result['error']})

        self._print_summary(results)
        return results

    def _print_summary(self, results):
        """Print a summary of all test results."""
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)

        successful_runs = [r for r in results if 'generated_proof' in r]
        failed_runs = [r for r in results if 'error' in r]
        print(f"Successful inferences: {len(successful_runs)}/{len(results)}")
        print(f"Failed inferences: {len(failed_runs)}/{len(results)}")

        if successful_runs:
            # --- ADDED: Verification stats ---
            verified_count = sum(1 for r in successful_runs if r['evaluation'].get('verified', False))
            total_time = successful_runs[0]['total_batch_time']
            exact_matches = sum(1 for r in successful_runs if r['evaluation']['exact_match'])
            avg_tactic_sim = sum(r['evaluation']['tactic_similarity'] for r in successful_runs) / len(successful_runs)
            
            print(f"Total batch inference time: {total_time:.2f}s")
            print(f"✅ Verified proofs: {verified_count}/{len(successful_runs)}")
            print(f"Exact text matches: {exact_matches}/{len(successful_runs)}")
            print(f"Average tactic similarity: {avg_tactic_sim:.2f}")

        print("\nIndividual Results:")
        for i, result in enumerate(results, 1):
            name = result['example']['name']
            status = "FAILED INFERENCE"
            if 'evaluation' in result:
                # --- ADDED: Verification status in summary line ---
                verified_status = "✅ VERIFIED" if result['evaluation'].get('verified', False) else "❌ UNVERIFIED"
                match = "EXACT" if result['evaluation']['exact_match'] else "PARTIAL"
                sim = result['evaluation']['tactic_similarity']
                status = f"{verified_status} | {match} (similarity: {sim:.2f})"
            print(f"  {i}. {name}: {status}")
        print("=" * 80)

    # --- Other methods (load_examples, create_prompt, etc.) are unchanged ---
    def load_herald_examples(self, num_examples: int = 3):
        print(f"Loading {num_examples} examples from Herald Proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas()
            df['formal_proof_len'] = df['formal_proof'].str.len()
            short_idx = df.index[df['formal_proof_len'] < 100][0] if not df[df['formal_proof_len'] < 100].empty else 0
            medium_idx = df.index[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)][0] if not df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)].empty else 1
            long_idx = df.index[df['formal_proof_len'] >= 300][0] if not df[df['formal_proof_len'] >= 300].empty else 2
            selected_indices = [short_idx, medium_idx, long_idx][:num_examples]
            examples = [df.iloc[idx].to_dict() for idx in selected_indices]
            for i, ex in enumerate(examples):
                print(f"  Example {i+1}: '{ex['name']}' (proof length: {len(ex['formal_proof'])} chars)")
            return examples
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    def create_prompt(self, example):
        return f"Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.\n\n{example['header']}\n\n{example['formal_theorem']} := by\n  sorry"
    def run_inference(self, prompts: list[str], max_steps: int = 1000):
        print(f"Running inference on a batch of {len(prompts)} prompts...")
        start_time = time.time()
        try:
            self.sampler.total_generation_steps = max_steps
            tokenized_prompts = [self.vocab.encode(p, add_bos=True) for p in prompts]
            max_len = max(len(t) for t in tokenized_prompts)
            padded_tokens = jnp.array([t + [self.vocab.pad_id()] * (max_len - len(t)) for t in tokenized_prompts])
            rng = jax.random.PRNGKey(0)
            output_tokens = self._jitted_sample_fn(self.params, rng, padded_tokens)
            generated_texts = self.vocab.decode(output_tokens.tolist())
            inference_time = time.time() - start_time
            return {'success': True, 'generated_texts': generated_texts, 'inference_time': inference_time}
        except Exception as e:
            return {'success': False, 'error': str(e), 'inference_time': time.time() - start_time}
    def extract_proof_from_output(self, output_text: str):
        if ':= by' in output_text:
            try:
                after_by = output_text.split(':= by', 1)[1]
                proof_lines = [line.strip() for line in after_by.split('\n') if line.strip() and 'sorry' not in line]
                if proof_lines: return '\n  '.join(proof_lines)
            except IndexError: pass
        return "No valid proof generated"
    def evaluate_example(self, generated_proof, ground_truth):
        generated_clean = generated_proof.replace(' ', '').replace('\n', '').lower()
        ground_truth_clean = ground_truth.replace(' ', '').replace('\n', '').lower()
        exact_match = generated_clean == ground_truth_clean
        tactics = ['rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction']
        ground_truth_tactics = {t for t in tactics if t in ground_truth.lower()}
        generated_tactics = {t for t in tactics if t in generated_proof.lower()}
        tactic_overlap = len(ground_truth_tactics.intersection(generated_tactics))
        tactic_total = len(ground_truth_tactics) if len(ground_truth_tactics) > 0 else 1
        return {'exact_match': exact_match, 'tactic_similarity': tactic_overlap / tactic_total}


def main():
    """Main execution function."""
    try:
        tester = HeraldInferenceTester()
        results = tester.run_test_suite()

        print("\nSave detailed results to file? (y/n): ", end="")
        if input().lower().startswith('y'):
            json_results = []
            for r in results:
                res = {
                    'example_name': str(r['example']['name']),
                    'example_id': int(r['example']['id'])
                }
                if 'error' in r:
                    res['error'] = r.get('error', 'Unknown error')
                else:
                    res.update({
                        'generated_proof': str(r['generated_proof']),
                        'exact_match': bool(r['evaluation']['exact_match']),
                        'tactic_similarity': float(r['evaluation']['tactic_similarity']),
                        # --- ADDED: Save verification status ---
                        'verified': bool(r['evaluation'].get('verified', False))
                    })
                json_results.append(res)

            with open('herald_inference_results_corrected.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            print("Results saved to 'herald_inference_results_corrected.json'")

        print("\nTest suite completed!")
        return 0
    except Exception as e:
        print(f"\nFatal error in main execution: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
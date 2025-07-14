# #!/usr/bin/env python3
# """
# Lean Proofs Model Inference & Verification Script
# (Correctly Refactored for Modularity and Data Flow)
# """

# import subprocess
# import time
# from pathlib import Path
# import traceback
# import argparse

# import jax
# import jax.numpy as jnp
# import numpy as np
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset
# from flax import jax_utils
# import tpu_profiler

# # Initialize JAX's distributed environment at the very beginning.
# jax.distributed.initialize()

# # This is a static function, making it cleaner to pmap.
# # It takes the replicated model parameters and a batch of sharded tokens.
# def pmapped_generate_from_tokens(params, tokenized_prompts, total_generation_steps):
#     # This is the core model's generate method, which expects tokens.
#     return rg.Griffin.generate(
#         tokenized_prompts,
#         params=params,
#         total_generation_steps=total_generation_steps,
#     )

# class RecurrentGemmaService:
#     def __init__(self, ckpt_dir: Path, tok_file: Path):
#         print(f"[Process {jax.process_index()}] Initializing RecurrentGemma model...")
#         if not ckpt_dir.is_dir():
#             raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
#         if not tok_file.is_file():
#             raise FileNotFoundError(f"Tokenizer file not found: {tok_file}")

#         self._load_model_and_setup_pmap(ckpt_dir, tok_file)
#         print(f"[Process {jax.process_index()}] Model and pmapped function ready!")

#     def _load_model_and_setup_pmap(self, ckpt_dir: Path, tok_file: Path):
#         restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
#         params = restored.get("params", restored)
#         self.replicated_params = jax_utils.replicate(params)
        
#         self.vocab = spm.SentencePieceProcessor(model_file=str(tok_file))

#         # pmap the static generation function.
#         self.generate_fn = jax.pmap(
#             pmapped_generate_from_tokens,
#             # in_axes corresponds to (params, prompts, steps)
#             in_axes=(0, 0, None),
#             static_broadcasted_argnums=(2,) # max_steps is static
#         )

#     def generate(self, prompts: list[str], max_steps: int) -> list[str]:
#         """
#         Handles the CPU-side work: tokenization and data preparation,
#         then calls the pmapped function.
#         """
#         # 1. Tokenize on CPU
#         tokenized_prompts = self.vocab.encode(prompts)
#         max_len = max(len(p) for p in tokenized_prompts)
#         padded_prompts = np.array(
#             [p + [self.vocab.pad_id()] * (max_len - len(p)) for p in tokenized_prompts]
#         )
        
#         # 2. Reshape for devices
#         num_devices = jax.local_device_count()
#         prompt_batch = padded_prompts.reshape((num_devices, -1, max_len))

#         # 3. Call the pmapped function with tokens
#         result_tokens = self.generate_fn(self.replicated_params, prompt_batch, max_steps)
#         result_tokens.block_until_ready()
        
#         # 4. Detokenize on CPU
#         result_tokens_flat = result_tokens.reshape(-1, result_tokens.shape[-1])
#         return self.vocab.decode(result_tokens_flat.tolist())

# class LeanVerifier:
#     def __init__(self, lean_project_path: Path):
#         self.lean_project_path = lean_project_path
#         self.lean_src_path = self.lean_project_path / "LeanVerifier"
#         if not self.lean_src_path.is_dir():
#             raise FileNotFoundError(f"Lean source directory not found: {self.lean_src_path}")

#     def verify(self, lean_code: str, proof_name: str) -> dict:
#         if ':= by' in lean_code:
#             proof_block = lean_code.split(':= by', 1)[-1]
#             if 'sorry' in proof_block:
#                 return {'verified': False, 'output': "Proof attempt used 'sorry'."}
#         else:
#              return {'verified': False, 'output': "Separator ':= by' not found."}

#         safe_filename = "".join(c if c.isalnum() else "_" for c in proof_name)
#         temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

#         try:
#             temp_lean_file.write_text(lean_code, encoding='utf-8')
#             proc = subprocess.run(
#                 ['lake', 'build'],
#                 cwd=self.lean_project_path,
#                 capture_output=True, text=True,
#                 timeout=120
#             )
#             return {
#                 'verified': proc.returncode == 0,
#                 'output': proc.stdout if proc.returncode == 0 else proc.stderr
#             }
#         except subprocess.TimeoutExpired:
#             return {'verified': False, 'output': 'Compiler verification timed out.'}
#         except Exception as e:
#             return {'verified': False, 'output': str(e)}
#         finally:
#             if temp_lean_file.exists():
#                 temp_lean_file.unlink()

# class HeraldTestSuite:
#     def __init__(self, gemma_service: RecurrentGemmaService, lean_verifier: LeanVerifier):
#         self.gemma = gemma_service
#         self.verifier = lean_verifier

#     def _load_data(self, num_examples: int) -> list[dict] | None:
#         if jax.process_index() != 0: return None
#         print(f"[Process 0] Preparing dataset for {num_examples} examples...")
#         try:
#             dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#             return dataset.to_pandas().sample(n=num_examples, random_state=42).to_dict(orient='records')
#         except Exception as e:
#             print(f"[Process 0] Error loading dataset: {e}")
#             return None

#     def _create_prompt(self, example: dict) -> str:
#         full_theorem = example.get('formal_theorem', '')
#         return full_theorem.split(':= by', 1)[0] + ":=" if ':= by' in full_theorem else full_theorem

#     def run(self, num_examples: int, max_steps: int):
#         if jax.process_index() != 0: return

#         print("\n" + "=" * 80 + "\nStarting Herald Proofs DISTRIBUTED Inference & Verification\n" + "=" * 80)
        
#         examples = self._load_data(num_examples)
#         if not examples:
#             print("No examples loaded. Exiting.")
#             return

#         prompts = [self._create_prompt(ex) for ex in examples]
        
#         print(f"[Process 0] Starting parallel inference with max_steps={max_steps}...")
#         start_time = time.time()
#         try:
#             generated_texts = self.gemma.generate(prompts, max_steps)
#             inference_time = time.time() - start_time
#             print(f"\nParallel inference completed in {inference_time:.2f}s")
            
#             results = self._verify_results(examples, generated_texts)
#             self._print_summary(results)

#         except Exception as e:
#             print(f"\n❌ An error occurred during inference: {e}")
#             traceback.print_exc()

#     def _verify_results(self, examples: list[dict], generated_texts: list[str]) -> list[dict]:
#         results_data = []
#         for i, full_code in enumerate(generated_texts):
#             example = examples[i]
#             print(f"\n--- Verifying EXAMPLE {i+1}/{len(examples)}: {example['name']} ---")
            
#             clean_code = full_code.replace(self.gemma.vocab.decode(self.gemma.vocab.pad_id()), "").strip()
            
#             verification = self.verifier.verify(clean_code, example['name'])
#             status_msg = "✅ Verification successful!" if verification['verified'] else f"❌ Verification failed: {verification['output'].strip().splitlines()[0]}"
#             print(status_msg)

#             results_data.append({'example': example['name'], 'verified': verification['verified']})
#         return results_data

#     def _print_summary(self, results: list[dict]):
#         print("\n" + "=" * 80 + "\nTEST SUITE SUMMARY\n" + "=" * 80)
#         if not results:
#             print("No results to summarize.")
#         else:
#             verified_runs = [r for r in results if r['verified']]
#             print(f"Total examples tested: {len(results)}")
#             print(f"✅ Successfully verified proofs: {len(verified_runs)}/{len(results)}")
#         print("=" * 80)

# def main():
#     parser = argparse.ArgumentParser(description="RecurrentGemma Lean Proof Inference & Verification")
#     script_dir = Path(__file__).parent.resolve()
    
#     parser.add_argument("--ckpt_dir", type=Path, default=script_dir / "2b/2b")
#     parser.add_argument("--tok_file", type=Path, default=script_dir / "2b/tokenizer.model")
#     parser.add_argument("--lean_project_path", type=Path, default=script_dir / "lean_verifier")
#     parser.add_argument("--num_examples", type=int, default=None)
#     parser.add_argument("--max_steps", type=int, default=1024)
#     args = parser.parse_args()

#     if args.num_examples is None:
#         args.num_examples = jax.device_count() or 1

#     try:
#         gemma_service = RecurrentGemmaService(ckpt_dir=args.ckpt_dir, tok_file=args.tok_file)
#         lean_verifier = LeanVerifier(lean_project_path=args.lean_project_path)
        
#         test_suite = HeraldTestSuite(gemma_service=gemma_service, lean_verifier=lean_verifier)
        
#         jax.block_until_ready(jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count())))
        
#         with tpu_profiler.profile():
#             test_suite.run(num_examples=args.num_examples, max_steps=args.max_steps)
        
#         if jax.process_index() == 0:
#             print("\nTest suite completed!")

#     except Exception as e:
#         print(f"\n🚨 A fatal error occurred on process {jax.process_index()}: {e}")
#         traceback.print_exc()
#         return 1
#     return 0

# if __name__ == "__main__":
#     exit(main())




#!/usr/bin/env python3
"""
Herald Proofs Model Inference Test (TPU v4-16 Optimized)

This script loads 3 examples from the Herald Proofs dataset and runs batched
inference using the RecurrentGemma model, optimized for a TPU v4-16 machine.
The script uses JAX's sharding capabilities to parallelize the workload
across all 16 devices.
"""

import json
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
        print("Initializing RecurrentGemma model for TPU v4-16")

        # File paths
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.tok_file = Path(tok_file).resolve()

        # Setup JAX device mesh for TPU v4-16 (16 devices)
        self.devices = jax.devices()
        if len(self.devices) != 16:
            print(f"⚠️ Warning: Expected 16 devices for a TPU v4-16, but found {len(self.devices)}. Performance may not be optimal.")

        # Create a 2D mesh, conventionally for data and model parallelism.
        # For batched inference, we will primarily use the 'data' axis.
        self.mesh = jsh.Mesh(self.devices, ('data',))
        print(f"JAX device mesh created with {self.mesh.size} devices.")

        # Load model and prepare for parallel execution
        self._load_model_and_prepare_jit()
        print("Model loaded and JIT-compiled successfully!")

    def _load_model_and_prepare_jit(self):
        """Load the RecurrentGemma model, shard parameters, and JIT-compile the generation function."""
        # Restore weights from checkpoint
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        params = restored.get("params", restored)

        # Configure model from preset
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

        # Initialize model and tokenizer
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

        # Define sharding within the mesh context.
        # For inference, we replicate the model parameters across all devices.
        with self.mesh:
            # Replicate parameters means every core gets a full copy of the model
            replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
            self.params = jax.device_put(params, replicated_sharding)

            # JIT-compile the model's generate function for efficient parallel execution.
            # This creates a highly optimized version of the function before it's ever called.
            self._jitted_generate = jax.jit(
                self.model.generate,
                # Specify how JAX should shard the function's arguments
                in_shardings=(
                    replicated_sharding,  # self (model parameters are replicated)
                    jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data')),  # Shard input_tokens on the 'data' axis
                    None,  # total_generation_steps (a static argument, no sharding needed)
                    replicated_sharding,  # params (also replicated)
                ),
                # The output will be sharded along the batch axis, just like the input
                out_shardings=jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data', None))
            )

    def load_herald_examples(self, num_examples: int = 3):
        """Load examples from Herald Proofs dataset."""
        print(f"Loading {num_examples} examples from Herald Proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas()
            df['formal_proof_len'] = df['formal_proof'].str.len()

            # Select a diverse set of examples based on proof length
            short_idx = df.index[df['formal_proof_len'] < 100][0] if not df[df['formal_proof_len'] < 100].empty else 0
            medium_idx = df.index[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)][0] if not df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)].empty else 1
            long_idx = df.index[df['formal_proof_len'] >= 300][0] if not df[df['formal_proof_len'] >= 300].empty else 2

            selected_indices = [short_idx, medium_idx, long_idx][:num_examples]
            examples = []
            for i, idx in enumerate(selected_indices):
                example = df.iloc[idx]
                examples.append({
                    'index': idx, 'id': example['id'], 'name': example['name'],
                    'header': example['header'], 'informal_theorem': example['informal_theorem'],
                    'formal_theorem': example['formal_theorem'], 'formal_proof': example['formal_proof'],
                    'informal_proof': example['informal_proof'], 'proof_length': len(example['formal_proof'])
                })
                print(f"  Example {i+1}: '{example['name']}' (proof length: {len(example['formal_proof'])} chars)")
            return examples
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def create_prompt(self, example):
        """Create a prompt for the model based on a Herald dataset example."""
        return f"Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.\n\n{example['header']}\n\n{example['formal_theorem']} := by\n  sorry"

    def run_inference(self, prompts: list[str], max_steps: int = 1000):
        """Run batched inference on a list of prompts."""
        print(f"Running inference on a batch of {len(prompts)} prompts...")
        start_time = time.time()

        try:
            # Tokenize and pad the batch of prompts to ensure they all have the same length
            tokenized_prompts = [self.vocab.encode(p) for p in prompts]
            max_len = max(len(t) for t in tokenized_prompts)
            padded_tokens = jnp.array([
                t + [self.vocab.pad_id()] * (max_len - len(t)) for t in tokenized_prompts
            ])

            # Run the JIT-compiled generation function on the entire batch
            output_tokens = self._jitted_generate(
                self.model,
                padded_tokens,
                total_generation_steps=max_steps,
                params=self.params,
            )

            # Decode the output tokens back to text. JAX handles gathering the sharded data back to the host.
            generated_texts = self.vocab.decode(output_tokens.tolist())
            inference_time = time.time() - start_time

            return {
                'success': True,
                'generated_texts': generated_texts,
                'inference_time': inference_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }

    def extract_proof_from_output(self, output_text: str):
        """Extract the proof portion from the model's generated output."""
        if ':= by' in output_text:
            try:
                # Get the text after the ':= by' keyword
                after_by = output_text.split(':= by', 1)[1]
                proof_lines = [
                    line.strip() for line in after_by.split('\n')
                    if line.strip() and 'sorry' not in line
                ]
                if proof_lines:
                    return '\n  '.join(proof_lines)
            except IndexError:
                pass
        return "No valid proof generated"

    def evaluate_example(self, generated_proof, ground_truth):
        """Perform a simple evaluation of the generated proof against the ground truth."""
        generated_clean = generated_proof.replace(' ', '').replace('\n', '').lower()
        ground_truth_clean = ground_truth.replace(' ', '').replace('\n', '').lower()
        exact_match = generated_clean == ground_truth_clean

        tactics = ['rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction']
        ground_truth_tactics = {t for t in tactics if t in ground_truth.lower()}
        generated_tactics = {t for t in tactics if t in generated_proof.lower()}

        tactic_overlap = len(ground_truth_tactics.intersection(generated_tactics))
        tactic_total = len(ground_truth_tactics) if ground_truth_tactics else 1

        return {
            'exact_match': exact_match,
            'tactic_similarity': tactic_overlap / tactic_total,
        }

    def run_test_suite(self):
        """Run the complete test suite on a batch of Herald examples."""
        print("Starting Herald Proofs inference test suite...")
        print("=" * 80)

        examples = self.load_herald_examples(3)
        if not examples:
            print("No examples loaded. Exiting.")
            return

        # Create a single batch of prompts from all examples
        prompts = [self.create_prompt(ex) for ex in examples]

        # Run inference on the entire batch in one go
        inference_result = self.run_inference(prompts)

        results = []
        if inference_result['success']:
            print(f"\nBatch inference completed in {inference_result['inference_time']:.2f}s")
            generated_texts = inference_result['generated_texts']

            for i, example in enumerate(examples):
                print(f"\nPROCESSING EXAMPLE {i+1}/{len(examples)}: {example['name']}")
                print("-" * 60)

                generated_text = generated_texts[i]
                generated_proof = self.extract_proof_from_output(generated_text)
                evaluation = self.evaluate_example(generated_proof, example['formal_proof'])

                print(f"Exact match: {evaluation['exact_match']}")
                print(f"Tactic similarity: {evaluation['tactic_similarity']:.2f}")
                print(f"\nGenerated Proof:\n--\n{generated_proof}\n--")
                print(f"\nGround Truth:\n--\n{example['formal_proof']}\n--")

                results.append({
                    'example': example, 'generated_text': generated_text,
                    'generated_proof': generated_proof, 'evaluation': evaluation,
                    'total_batch_time': inference_result['inference_time']
                })
        else:
            print(f"Batch inference failed: {inference_result['error']}")
            for example in examples:
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
            total_time = successful_runs[0]['total_batch_time']
            exact_matches = sum(1 for r in successful_runs if r['evaluation']['exact_match'])
            avg_tactic_sim = sum(r['evaluation']['tactic_similarity'] for r in successful_runs) / len(successful_runs)

            print(f"Total batch inference time: {total_time:.2f}s")
            print(f"Exact matches: {exact_matches}/{len(successful_runs)}")
            print(f"Average tactic similarity: {avg_tactic_sim:.2f}")

        print("\nIndividual Results:")
        for i, result in enumerate(results, 1):
            name = result['example']['name']
            status = "FAILED"
            if 'evaluation' in result:
                match = "EXACT" if result['evaluation']['exact_match'] else "PARTIAL"
                sim = result['evaluation']['tactic_similarity']
                status = f"{match} (similarity: {sim:.2f})"
            print(f"  {i}. {name}: {status}")
        print("=" * 80)

def main():
    """Main execution function."""
    try:
        tester = HeraldInferenceTester()
        results = tester.run_test_suite()

        print("\nSave detailed results to file? (y/n): ", end="")
        if input().lower().startswith('y'):
            json_results = []
            for r in results:
                res = {'example_name': r['example']['name'], 'example_id': r['example']['id']}
                if 'error' in r:
                    res['error'] = r.get('error', 'Unknown error')
                else:
                    res.update({
                        'generated_proof': r['generated_proof'],
                        'exact_match': r['evaluation']['exact_match'],
                        'tactic_similarity': r['evaluation']['tactic_similarity']
                    })
                json_results.append(res)

            with open('herald_inference_results_v4-16.json', 'w') as f:
                json.dump(json_results, f, indent=2)
            print("Results saved to 'herald_inference_results_v4-16.json'")

        print("\nTest suite completed!")
        return 0
    except Exception as e:
        print(f"\nFatal error in main execution: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
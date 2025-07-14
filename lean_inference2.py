

# #!/usr/bin/env python3
# """
# Herald Proofs Model Inference Test (TPU v4-16 Optimized)

# This script loads 3 examples from the Herald Proofs dataset and runs batched
# inference using the RecurrentGemma model, optimized for a TPU v4-16 machine.
# The script uses JAX's sharding capabilities to parallelize the workload
# across all 16 devices.
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
#         print("Initializing RecurrentGemma model for TPU v4-16")

#         # File paths
#         self.ckpt_dir = Path(ckpt_dir).resolve()
#         self.tok_file = Path(tok_file).resolve()

#         # Setup JAX device mesh for TPU v4-16 (16 devices)
#         self.devices = jax.devices()
#         if len(self.devices) != 16:
#             print(f"⚠️ Warning: Expected 16 devices for a TPU v4-16, but found {len(self.devices)}. Performance may not be optimal.")

#         # Create a 2D mesh, conventionally for data and model parallelism.
#         # For batched inference, we will primarily use the 'data' axis.
#         self.mesh = jsh.Mesh(self.devices, ('data',))
#         print(f"JAX device mesh created with {self.mesh.size} devices.")

#         # Load model and prepare for parallel execution
#         self._load_model_and_prepare_jit()
#         print("Model loaded and JIT-compiled successfully!")

#     def _load_model_and_prepare_jit(self):
#         """Load the RecurrentGemma model, shard parameters, and JIT-compile the generation function."""
#         # Restore weights from checkpoint
#         restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
#         params = restored.get("params", restored)

#         # Configure model from preset
#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

#         # Initialize model and tokenizer
#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

#         # Define sharding within the mesh context.
#         # For inference, we replicate the model parameters across all devices.
#         with self.mesh:
#             # Replicate parameters means every core gets a full copy of the model
#             replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
#             self.params = jax.device_put(params, replicated_sharding)

#             # JIT-compile the model's generate function for efficient parallel execution.
#             # This creates a highly optimized version of the function before it's ever called.
#             self._jitted_generate = jax.jit(
#                 self.model.generate,
#                 # Specify how JAX should shard the function's arguments
#                 in_shardings=(
#                     replicated_sharding,  # self (model parameters are replicated)
#                     jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data')),  # Shard input_tokens on the 'data' axis
#                     None,  # total_generation_steps (a static argument, no sharding needed)
#                     replicated_sharding,  # params (also replicated)
#                 ),
#                 # The output will be sharded along the batch axis, just like the input
#                 out_shardings=jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data', None))
#             )

#     def load_herald_examples(self, num_examples: int = 3):
#         """Load examples from Herald Proofs dataset."""
#         print(f"Loading {num_examples} examples from Herald Proofs dataset...")
#         try:
#             dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#             df = dataset.to_pandas()
#             df['formal_proof_len'] = df['formal_proof'].str.len()

#             # Select a diverse set of examples based on proof length
#             short_idx = df.index[df['formal_proof_len'] < 100][0] if not df[df['formal_proof_len'] < 100].empty else 0
#             medium_idx = df.index[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)][0] if not df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)].empty else 1
#             long_idx = df.index[df['formal_proof_len'] >= 300][0] if not df[df['formal_proof_len'] >= 300].empty else 2

#             selected_indices = [short_idx, medium_idx, long_idx][:num_examples]
#             examples = []
#             for i, idx in enumerate(selected_indices):
#                 example = df.iloc[idx]
#                 examples.append({
#                     'index': idx, 'id': example['id'], 'name': example['name'],
#                     'header': example['header'], 'informal_theorem': example['informal_theorem'],
#                     'formal_theorem': example['formal_theorem'], 'formal_proof': example['formal_proof'],
#                     'informal_proof': example['informal_proof'], 'proof_length': len(example['formal_proof'])
#                 })
#                 print(f"  Example {i+1}: '{example['name']}' (proof length: {len(example['formal_proof'])} chars)")
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
#             # Tokenize and pad the batch of prompts to ensure they all have the same length
#             tokenized_prompts = [self.vocab.encode(p) for p in prompts]
#             max_len = max(len(t) for t in tokenized_prompts)
#             padded_tokens = jnp.array([
#                 t + [self.vocab.pad_id()] * (max_len - len(t)) for t in tokenized_prompts
#             ])

#             # Run the JIT-compiled generation function on the entire batch
#             output_tokens = self._jitted_generate(
#                 self.model,
#                 padded_tokens,
#                 total_generation_steps=max_steps,
#                 params=self.params,
#             )

#             # Decode the output tokens back to text. JAX handles gathering the sharded data back to the host.
#             generated_texts = self.vocab.decode(output_tokens.tolist())
#             inference_time = time.time() - start_time

#             return {
#                 'success': True,
#                 'generated_texts': generated_texts,
#                 'inference_time': inference_time
#             }
#         except Exception as e:
#             return {
#                 'success': False,
#                 'error': str(e),
#                 'inference_time': time.time() - start_time
#             }

#     def extract_proof_from_output(self, output_text: str):
#         """Extract the proof portion from the model's generated output."""
#         if ':= by' in output_text:
#             try:
#                 # Get the text after the ':= by' keyword
#                 after_by = output_text.split(':= by', 1)[1]
#                 proof_lines = [
#                     line.strip() for line in after_by.split('\n')
#                     if line.strip() and 'sorry' not in line
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
#         tactic_total = len(ground_truth_tactics) if ground_truth_tactics else 1

#         return {
#             'exact_match': exact_match,
#             'tactic_similarity': tactic_overlap / tactic_total,
#         }

#     def run_test_suite(self):
#         """Run the complete test suite on a batch of Herald examples."""
#         print("Starting Herald Proofs inference test suite...")
#         print("=" * 80)

#         examples = self.load_herald_examples(3)
#         if not examples:
#             print("No examples loaded. Exiting.")
#             return

#         # Create a single batch of prompts from all examples
#         prompts = [self.create_prompt(ex) for ex in examples]

#         # Run inference on the entire batch in one go
#         inference_result = self.run_inference(prompts)

#         results = []
#         if inference_result['success']:
#             print(f"\nBatch inference completed in {inference_result['inference_time']:.2f}s")
#             generated_texts = inference_result['generated_texts']

#             for i, example in enumerate(examples):
#                 print(f"\nPROCESSING EXAMPLE {i+1}/{len(examples)}: {example['name']}")
#                 print("-" * 60)

#                 generated_text = generated_texts[i]
#                 generated_proof = self.extract_proof_from_output(generated_text)
#                 evaluation = self.evaluate_example(generated_proof, example['formal_proof'])

#                 print(f"Exact match: {evaluation['exact_match']}")
#                 print(f"Tactic similarity: {evaluation['tactic_similarity']:.2f}")
#                 print(f"\nGenerated Proof:\n--\n{generated_proof}\n--")
#                 print(f"\nGround Truth:\n--\n{example['formal_proof']}\n--")

#                 results.append({
#                     'example': example, 'generated_text': generated_text,
#                     'generated_proof': generated_proof, 'evaluation': evaluation,
#                     'total_batch_time': inference_result['inference_time']
#                 })
#         else:
#             print(f"Batch inference failed: {inference_result['error']}")
#             for example in examples:
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
#                 res = {'example_name': r['example']['name'], 'example_id': r['example']['id']}
#                 if 'error' in r:
#                     res['error'] = r.get('error', 'Unknown error')
#                 else:
#                     res.update({
#                         'generated_proof': r['generated_proof'],
#                         'exact_match': r['evaluation']['exact_match'],
#                         'tactic_similarity': r['evaluation']['tactic_similarity']
#                     })
#                 json_results.append(res)

#             with open('herald_inference_results_v4-16.json', 'w') as f:
#                 json.dump(json_results, f, indent=2)
#             print("Results saved to 'herald_inference_results_v4-16.json'")

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

This script loads 3 examples from the Herald Proofs dataset and runs batched
inference using the RecurrentGemma model, optimized for a TPU v4-16 machine.
The script uses JAX's sharding capabilities to parallelize the workload
across all 16 devices.
"""

import json
import time
from pathlib import Path
from functools import partial

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

    def __init__(self, ckpt_dir: str = "2b-it/2b-it", tok_file: str = "2b-it/tokenizer.model"):
        """Initialize the model, tokenizer, and JAX device mesh."""
        print("Initializing RecurrentGemma model... 🚀")

        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.tok_file = Path(tok_file).resolve()

        self.devices = jax.devices()
        print(f"JAX device mesh created with {len(self.devices)} devices.")
        
        # A 1D mesh is flexible and works for any number of devices.
        self.mesh = jsh.Mesh(self.devices, ('data',))

        self._load_model_and_prepare_jit()
        print("Model loaded and JIT-compiled successfully!")

    def _load_model_and_prepare_jit(self):
        """Load the RecurrentGemma model, shard parameters, and JIT-compile the generation function."""
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        params = restored.get("params", restored)

        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)

        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        
        # The Sampler class knows how to correctly run the generation loop.
        # We create it to get access to its internal, JIT-compiled 'sample_fn'.
        self.sampler = rg.Sampler(
            model=self.model,
            vocab=self.vocab,
            params=params, # Pass unsharded params initially
        )
        
        with self.mesh:
            # Replicate parameters means every core gets a full copy of the model
            replicated_sharding = jsh.NamedSharding(self.mesh, jsh.PartitionSpec())
            self.params = jax.device_put(params, replicated_sharding)

            # We now JIT the sampler's internal function, which is designed for numerical inputs.
            # This allows us to apply our batching and sharding strategy correctly.
            self._jitted_sample_fn = jax.jit(
                self.sampler.sample_fn,
                # Sharding for inputs: (params, rng, input_tokens)
                in_shardings=(
                    replicated_sharding,
                    None, # RNG key is not sharded
                    jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data')), # Shard tokens on the 'data' axis
                ),
                out_shardings=jsh.NamedSharding(self.mesh, jsh.PartitionSpec('data'))
            )

    def load_herald_examples(self, num_examples: int = 3):
        """Load examples from Herald Proofs dataset."""
        print(f"Loading {num_examples} examples from Herald Proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas()
            df['formal_proof_len'] = df['formal_proof'].str.len()

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
            # The sampler's __call__ method sets the total generation steps.
            # This is a bit of a workaround to configure the internal static loop length.
            self.sampler.total_generation_steps = max_steps

            # Tokenize and pad the batch of prompts
            tokenized_prompts = [self.vocab.encode(p, add_bos=True) for p in prompts]
            max_len = max(len(t) for t in tokenized_prompts)
            padded_tokens = jnp.array([
                t + [self.vocab.pad_id()] * (max_len - len(t)) for t in tokenized_prompts
            ])

            # Run the JIT-compiled generation function
            rng = jax.random.PRNGKey(0)
            output_tokens = self._jitted_sample_fn(
                self.params,
                rng,
                padded_tokens,
            )

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

        prompts = [self.create_prompt(ex) for ex in examples]
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
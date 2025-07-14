# #!/usr/bin/env python3
# """
# Lean Proofs Model Inference & Verification Script
# (Refactored for Multi-Worker TPU Execution)
# """

# import subprocess
# import time
# from pathlib import Path

# import jax
# import jax.numpy as jnp
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset
# from flax import jax_utils
# import tpu_profiler
# # Import the utility for broadcasting data across hosts
# from jax.experimental import multihost_utils

# # Initialize JAX's distributed environment at the very beginning.
# jax.distributed.initialize()

# # Get the absolute path of the directory containing this script.
# SCRIPT_DIR = Path(__file__).parent.resolve()


# class HeraldInferenceTester:
#     """
#     Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
#     """

#     def __init__(self):
#         """Initialize the model and tokenizer."""
#         print(f"[Process {jax.process_index()}] Initializing RecurrentGemma model...")

#         self.repo_root = SCRIPT_DIR
#         self.ckpt_dir = self.repo_root / "2b" / "2b"
#         self.tok_file = self.repo_root / "2b" / "tokenizer.model"
#         self.lean_project_path = self.repo_root / "lean_verifier"
#         self.lean_src_path = self.lean_project_path / "LeanVerifier"

#         if not self.ckpt_dir.exists():
#             raise FileNotFoundError(f"Checkpoint directory not found at: {self.ckpt_dir}")
#         if not self.tok_file.exists():
#             raise FileNotFoundError(f"Tokenizer file not found at: {self.tok_file}")
#         if not self.lean_src_path.exists():
#             raise FileNotFoundError(f"Lean source directory not found at: {self.lean_src_path}")

#         self._load_model_and_sampler()
#         print(f"[Process {jax.process_index()}] Model loaded successfully!")

#     def _load_model_and_sampler(self):
#         """Load model, params, and create a pmap'd sampler for parallel inference."""
#         restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
#         self.params = restored.get("params", restored)
        
#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

#         self.replicated_params = jax_utils.replicate(self.params)

#         sampler = rg.Sampler(
#             model=self.model,
#             vocab=self.vocab,
#             params=self.replicated_params,
#             deterministic_sampling=True,
#             is_it_model=False # Correctly set for a base model
#         )

#         self.pmapped_sampler = jax.pmap(sampler, in_axes=(0), out_axes=0)

#     def load_herald_examples(self, num_examples: int = 8):
#         """Load examples. Process 0 downloads and broadcasts to all."""
#         examples = None
#         if jax.process_index() == 0:
#             print(f"[Process 0] Preparing dataset...")
#             if num_examples % jax.device_count() != 0:
#                 raise ValueError(
#                     f"Number of examples ({num_examples}) must be a multiple of "
#                     f"the number of devices ({jax.device_count()})."
#                 )
#             try:
#                 dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#                 df = dataset.to_pandas().sample(frac=1).reset_index(drop=True)
#                 examples = [row.to_dict() for _, row in df.head(num_examples).iterrows()]
#                 print(f"  [Process 0] Loaded {len(examples)} examples.")
#             except Exception as e:
#                 print(f"[Process 0] Error loading dataset: {e}")
#                 examples = [] # Send empty list on error
        
#         # Broadcast the data from process 0 to all other processes.
#         # This ensures every worker has the same list of examples.
#         examples = multihost_utils.broadcast_one_to_all(examples)
#         return examples

#     def create_prompt(self, example: dict) -> str:
#         """
#         Create a simple completion prompt suitable for a base model.
#         It finds the ':= by' and provides only the text before it.
#         """
#         full_theorem = example.get('formal_theorem', '')
        
#         try:
#             proof_start_index = full_theorem.index(':= by')
#             prompt_text = full_theorem[:proof_start_index].strip()
#             return prompt_text
#         except ValueError:
#             return full_theorem

#     def run_inference_parallel(self, prompts: list, max_steps: int = 1000) -> dict:
#         """Run inference in parallel on a BATCH of prompts."""
#         print(f"[Process {jax.process_index()}] Starting parallel inference...")
#         start_time = time.time()
        
#         try:
#             num_devices = jax.local_device_count()
#             prompt_batch = jnp.array(prompts).reshape((num_devices, -1))

#             result = self.pmapped_sampler(
#                 prompt_batch,
#                 total_generation_steps=max_steps
#             )
#             result.block_until_ready()
#             inference_time = time.time() - start_time
            
#             generated_texts = result.text.flatten().tolist()
            
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

#     def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
#         """Verification logic. Runs only on process 0."""
#         proof_block = full_code.split(':= by', 1)[-1]
#         if 'sorry' in proof_block:
#             print("❌ Verification failed: Model used the 'sorry' tactic.")
#             return {'verified': False, 'output': "Proof attempt used 'sorry'."}

#         safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
#         temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

#         try:
#             temp_lean_file.write_text(full_code, encoding='utf-8')
#             proc = subprocess.run(
#                 ['lake', 'build'],
#                 cwd=self.lean_project_path,
#                 capture_output=True, text=True, timeout=120
#             )
#             if proc.returncode == 0:
#                 print("✅ Verification successful!")
#                 return {'verified': True, 'output': proc.stdout}
#             else:
#                 print("❌ Verification failed: Compilation errors.")
#                 return {'verified': False, 'output': proc.stderr}
#         except Exception as e:
#             return {'verified': False, 'output': str(e)}
#         finally:
#             if temp_lean_file.exists():
#                 temp_lean_file.unlink()

#     def run_test_suite(self):
#         """Run the complete distributed test suite."""
#         if jax.process_index() == 0:
#             print("\n" + "=" * 80)
#             print("Starting Herald Proofs DISTRIBUTED Inference & Verification")
#             print("=" * 80)

#         # 1. All processes call this function. Process 0 loads and broadcasts data.
#         examples = self.load_herald_examples(num_examples=jax.device_count())
        
#         if not examples:
#             if jax.process_index() == 0:
#                 print("No examples were loaded or broadcasted. Exiting.")
#             return

#         # 2. All processes now have the data and can create prompts.
#         prompts = [self.create_prompt(ex) for ex in examples]
        
#         # 3. All processes must participate in the pmap call.
#         inference_result = self.run_inference_parallel(prompts)
        
#         # 4. Verification and summarization happens only on process 0.
#         if jax.process_index() == 0:
#             print(f"\nParallel inference completed in {inference_result['inference_time']:.2f}s")
            
#             results_data = []
#             if inference_result['success']:
#                 for i, generated_text in enumerate(inference_result['generated_texts']):
#                     full_generated_code = generated_text
#                     example = examples[i]
#                     print(f"\n--- Verifying EXAMPLE {i+1}/{len(examples)}: {example['name']} ---")
#                     verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])
#                     results_data.append({
#                         'example': example,
#                         'verified': verification_result['verified'],
#                     })
#             else:
#                 print(f"Inference failed: {inference_result['error']}")

#             self._print_summary(results_data)

#     def _print_summary(self, results: list):
#         """Print a final summary (only on process 0)."""
#         print("\n" + "=" * 80)
#         print("TEST SUITE SUMMARY")
#         print("=" * 80)

#         verified_runs = [r for r in results if r.get('verified')]
#         print(f"Total examples tested: {len(results)}")
#         print(f"Successfully generated and verified proofs: {len(verified_runs)}/{len(results)}")
#         print("=" * 80)

# def main():
#     """Main execution function for the script."""
#     try:
#         with tpu_profiler.profile():
#             tester = HeraldInferenceTester()
#             tester.run_test_suite()

#         # Barrier to ensure all processes finish before the script exits.
#         jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count()))
        
#         if jax.process_index() == 0:
#             print("\nTest suite completed!")

#     except Exception as e:
#         print(f"\nA fatal error occurred in main on process {jax.process_index()}: {e}")
#         import traceback
#         traceback.print_exc()
#         return 1
#     return 0


# if __name__ == "__main__":
#     exit(main())




#!/usr/bin/env python3
"""
Lean Proofs Model Inference & Verification Script
(Refactored for Multi-Worker TPU Execution & API Compatibility)
"""

import subprocess
import time
from pathlib import Path
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset
from flax import jax_utils
import tpu_profiler

# Initialize JAX's distributed environment at the very beginning.
jax.distributed.initialize()

# Get the absolute path of the directory containing this script.
# Assumes the script is in the root of the cloned repository.
SCRIPT_DIR = Path(__file__).parent.resolve()


class HeraldInferenceTester:
    """
    Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
    """

    def __init__(self):
        """Initialize the model and tokenizer for the current JAX process."""
        print(f"[Process {jax.process_index()}] Initializing RecurrentGemma model...")

        self.repo_root = SCRIPT_DIR
        # Define paths relative to the script's location
        self.ckpt_dir = self.repo_root / "2b" / "2b"
        self.tok_file = self.repo_root / "2b" / "tokenizer.model"
        self.lean_project_path = self.repo_root / "lean_verifier"
        self.lean_src_path = self.lean_project_path / "LeanVerifier"

        # Check for necessary files before proceeding
        if not self.ckpt_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found at: {self.ckpt_dir}")
        if not self.tok_file.is_file():
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tok_file}")
        if not self.lean_src_path.is_dir():
            raise FileNotFoundError(f"Lean source directory not found at: {self.lean_src_path}")

        self._load_model_and_sampler()
        print(f"[Process {jax.process_index()}] Model and sampler loaded successfully!")

    def _load_model_and_sampler(self):
        """Load model, params, and create a pmap'd sampler for parallel inference."""
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        self.params = restored.get("params", restored)
        
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

        # API FIX: Use the 'Sampler' class for generation. The 'Griffin' model
        # itself does not have a '.generate()' method.
        self.sampler = rg.Sampler(model=self.model, params=self.params, vocab=self.vocab)

        self.replicated_params = jax_utils.replicate(self.params)

        def generate_fn(params, tokenized_prompts, total_generation_steps):
            # Call the sampler instance, which handles the autoregressive loop.
            return self.sampler(
                input_tokens=tokenized_prompts,
                params=params,
                total_generation_steps=total_generation_steps,
                is_it_model=False
            )

        # pmap the generation function.
        self.pmapped_generate = jax.pmap(
            generate_fn,
            # in_axes specifies how to map arguments to devices:
            #   0: Shard the argument along its first axis.
            #   None: Broadcast the argument to all devices.
            in_axes=(0, 0, None),
        )

    def load_herald_examples(self, num_examples: int):
        """Load examples. Only process 0 downloads and prepares the data."""
        if jax.process_index() != 0:
            return None

        print(f"[Process 0] Preparing dataset...")
        num_devices = jax.device_count()
        if num_devices > 0 and num_examples % num_devices != 0:
            raise ValueError(
                f"Number of examples ({num_examples}) must be a multiple of "
                f"the number of devices ({num_devices})."
            )
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas().sample(n=num_examples, random_state=42).reset_index(drop=True)
            examples = df.to_dict(orient='records')
            print(f"  [Process 0] Loaded {len(examples)} examples.")
            return examples
        except Exception as e:
            print(f"[Process 0] Error loading dataset: {e}")
            return None

    def create_prompt(self, example: dict) -> str:
        """Create a simple completion prompt by stripping the proof from the theorem."""
        full_theorem = example.get('formal_theorem', '')
        try:
            # The prompt is the theorem statement up to ':= by'
            proof_start_index = full_theorem.index(':= by')
            return full_theorem[:proof_start_index] + ":="
        except ValueError:
            return full_theorem

    def run_inference_parallel(self, prompts: list, max_steps: int = 1024) -> dict:
        """Tokenize prompts on CPU, then run inference in parallel on TPUs."""
        print(f"[Process 0] Starting parallel inference...")
        start_time = time.time()
        
        try:
            # 1. Tokenize all prompts on the host CPU (process 0)
            tokenized_prompts = self.vocab.encode(prompts)

            # 2. Pad to the same length
            max_len = max(len(p) for p in tokenized_prompts)
            padded_prompts = np.array(
                [p + [self.vocab.pad_id()] * (max_len - len(p)) for p in tokenized_prompts]
            )
            
            # 3. Reshape for pmap (num_devices, prompts_per_device, sequence_length)
            num_devices = jax.local_device_count()
            prompt_batch = padded_prompts.reshape((num_devices, -1, max_len))

            # 4. Run the pmapped generation function
            result_tokens = self.pmapped_generate(
                self.replicated_params,
                prompt_batch,
                max_steps
            )
            result_tokens.block_until_ready()
            inference_time = time.time() - start_time
            
            # 5. Detokenize the results back to strings
            result_tokens_flat = result_tokens.reshape(-1, result_tokens.shape[-1])
            generated_texts = self.vocab.decode(result_tokens_flat.tolist())
            
            return {
                'success': True,
                'generated_texts': generated_texts,
                'inference_time': inference_time
            }
        except Exception as e:
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }

    def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
        """Verify a generated Lean proof by compiling it. Runs only on process 0."""
        # Clean up the generated text
        full_code = full_code.replace(self.vocab.decode(self.vocab.pad_id()), "").strip()

        if ':= by' not in full_code:
            print(f"❌ Verification failed: ':= by' separator not found.")
            return {'verified': False, 'output': "Separator ':= by' not found."}

        proof_block = full_code.split(':= by', 1)[-1]
        if 'sorry' in proof_block:
            print("❌ Verification failed: Model used the 'sorry' tactic.")
            return {'verified': False, 'output': "Proof attempt used 'sorry'."}

        safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
        temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

        try:
            temp_lean_file.write_text(full_code, encoding='utf-8')
            proc = subprocess.run(
                ['lake', 'build'],
                cwd=self.lean_project_path,
                capture_output=True, text=True, timeout=120
            )
            if proc.returncode == 0:
                print("✅ Verification successful!")
                return {'verified': True, 'output': proc.stdout}
            else:
                print("❌ Verification failed: Compilation errors.")
                return {'verified': False, 'output': proc.stderr}
        except Exception as e:
            return {'verified': False, 'output': str(e)}
        finally:
            if temp_lean_file.exists():
                temp_lean_file.unlink()

    def run_test_suite(self):
        """Run the complete distributed test suite. Orchestrated by process 0."""
        if jax.process_index() != 0:
            return

        print("\n" + "=" * 80)
        print("Starting Herald Proofs DISTRIBUTED Inference & Verification")
        print("=" * 80)
        
        num_devices = jax.device_count()
        if num_devices == 0:
            print("⚠️ No JAX devices found. Running on CPU with 1 example.")
            num_devices = 1

        examples = self.load_herald_examples(num_examples=num_devices)
        if not examples:
            print("No examples loaded. Exiting.")
            return

        prompts = [self.create_prompt(ex) for ex in examples]
        
        inference_result = self.run_inference_parallel(prompts)
        
        print(f"\nParallel inference completed in {inference_result.get('inference_time', 0):.2f}s")
        
        results_data = []
        if inference_result['success']:
            for i, full_generated_code in enumerate(inference_result['generated_texts']):
                example = examples[i]
                print(f"\n--- Verifying EXAMPLE {i+1}/{len(examples)}: {example['name']} ---")
                verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])
                results_data.append({
                    'example': example['name'],
                    'verified': verification_result['verified'],
                })
        else:
            print(f"\n❌ Inference failed: {inference_result['error']}")

        self._print_summary(results_data)

    def _print_summary(self, results: list):
        """Print a final summary. Runs only on process 0."""
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)

        if not results:
            print("No results to summarize.")
        else:
            verified_runs = [r for r in results if r.get('verified')]
            print(f"Total examples tested: {len(results)}")
            print(f"✅ Successfully verified proofs: {len(verified_runs)}/{len(results)}")
        print("=" * 80)

def main():
    """Main execution function for the script."""
    try:
        # All processes initialize the model and load it to their devices.
        tester = HeraldInferenceTester()
        
        # Barrier to ensure all processes finish initialization before proceeding.
        jax.block_until_ready(jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count())))
        
        # Process 0 orchestrates the test suite and profiling.
        if jax.process_index() == 0:
            with tpu_profiler.profile():
                tester.run_test_suite()
            print("\nTest suite completed!")

    except Exception as e:
        print(f"\n🚨 A fatal error occurred on process {jax.process_index()}: {e}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
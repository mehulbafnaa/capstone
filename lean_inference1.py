
# #!/usr/bin/env python3
# """
# Lean Proofs Model Inference & Verification Script
# (Refactored for Multi-Worker TPU Execution)

# This script loads examples from the Herald Proofs dataset, runs inference
# using a RecurrentGemma model in parallel across multiple TPU workers using JAX,
# and then uses the Lean 4 compiler to formally verify the correctness of the
# generated proof on a single host.
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

# # Initialize JAX's distributed environment at the very beginning.
# jax.distributed.initialize()

# # Get the absolute path of the directory containing this script.
# SCRIPT_DIR = Path(__file__).parent.resolve()


# class HeraldInferenceTester:
#     """
#     Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
#     This class is configured for multi-worker TPU execution.
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
#             is_it_model=True
#         )

#         self.pmapped_sampler = jax.pmap(sampler, in_axes=(0), out_axes=0)

#     def load_herald_examples(self, num_examples: int = 8):
#         """Load examples. Process 0 downloads and broadcasts to all."""
#         print(f"[Process {jax.process_index()}] Preparing dataset...")
        
#         if num_examples % jax.device_count() != 0:
#             raise ValueError(
#                 f"Number of examples ({num_examples}) must be a multiple of "
#                 f"the number of devices ({jax.device_count()})."
#             )

#         examples = []
#         if jax.process_index() == 0:
#             try:
#                 dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
#                 df = dataset.to_pandas().sample(frac=1).reset_index(drop=True)
#                 examples_data = df.head(num_examples)
                
#                 for _, row in examples_data.iterrows():
#                     examples.append(row.to_dict())
#                 print(f"  [Process 0] Loaded {len(examples)} examples.")
#             except Exception as e:
#                 print(f"[Process 0] Error loading dataset: {e}")
        
#         # Broadcast the data from process 0 to all other processes.
#         # This ensures every worker has the same list of examples.
#         examples = jax_utils.broadcast_data_across_hosts(examples)
#         return list(examples)

#     def create_prompt(self, example: dict) -> str:
#         """Create a standardized prompt for the model."""
#         return f"""Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.

# {example['header']}

# {example['formal_theorem']} := by
#   sorry"""

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
        
#         # 1. All processes get the same full list of examples.
#         examples = self.load_herald_examples(num_examples=jax.device_count())
        
#         if not examples:
#             if jax.process_index() == 0:
#                 print("No examples loaded. Exiting.")
#             return

#         # 2. All processes create the prompts for all examples.
#         prompts = [self.create_prompt(ex) for ex in examples]

#         # 3. All processes must call the pmapped function.
#         inference_result = self.run_inference_parallel(prompts)
        
#         # 4. Verification and summarization happens only on process 0.
#         if jax.process_index() == 0:
#             print(f"\nParallel inference completed in {inference_result['inference_time']:.2f}s")
            
#             results_data = []
#             if inference_result['success']:
#                 for i, generated_text in enumerate(inference_result['generated_texts']):
#                     example = examples[i]
#                     print(f"\n--- Verifying EXAMPLE {i+1}/{len(examples)}: {example['name']} ---")
#                     verification_result = self.verify_with_lean_compiler(generated_text, example['name'])
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
#         tester = HeraldInferenceTester()
#         tester.run_test_suite()
#         # Barrier ensures all processes finish before the script exits.
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
(Refactored for Multi-Worker TPU Execution)
"""

import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset
from flax import jax_utils

# Initialize JAX's distributed environment at the very beginning.
jax.distributed.initialize()

# Get the absolute path of the directory containing this script.
SCRIPT_DIR = Path(__file__).parent.resolve()


class HeraldInferenceTester:
    """
    Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
    """

    def __init__(self):
        """Initialize the model and tokenizer."""
        print(f"[Process {jax.process_index()}] Initializing RecurrentGemma model...")

        self.repo_root = SCRIPT_DIR
        self.ckpt_dir = self.repo_root / "2b" / "2b"
        self.tok_file = self.repo_root / "2b" / "tokenizer.model"
        self.lean_project_path = self.repo_root / "lean_verifier"
        self.lean_src_path = self.lean_project_path / "LeanVerifier"

        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found at: {self.ckpt_dir}")
        if not self.tok_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tok_file}")
        if not self.lean_src_path.exists():
            raise FileNotFoundError(f"Lean source directory not found at: {self.lean_src_path}")

        self._load_model_and_sampler()
        print(f"[Process {jax.process_index()}] Model loaded successfully!")

    def _load_model_and_sampler(self):
        """Load model, params, and create a pmap'd sampler for parallel inference."""
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        self.params = restored.get("params", restored)
        
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

        self.replicated_params = jax_utils.replicate(self.params)

        sampler = rg.Sampler(
            model=self.model,
            vocab=self.vocab,
            params=self.replicated_params,
            deterministic_sampling=True,
            is_it_model=False
        )

        self.pmapped_sampler = jax.pmap(sampler, in_axes=(0), out_axes=0)

    def load_herald_examples(self, num_examples: int = 8):
        """Load examples. Only process 0 downloads the data."""
        if jax.process_index() != 0:
            return [] # Other processes return an empty list

        print(f"[Process 0] Preparing dataset...")
        
        if num_examples % jax.device_count() != 0:
            raise ValueError(
                f"Number of examples ({num_examples}) must be a multiple of "
                f"the number of devices ({jax.device_count()})."
            )

        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas().sample(frac=1).reset_index(drop=True)
            examples = [row.to_dict() for _, row in df.head(num_examples).iterrows()]
            print(f"  [Process 0] Loaded {len(examples)} examples.")
            return examples
        except Exception as e:
            print(f"[Process 0] Error loading dataset: {e}")
            return []

    def create_prompt(self, example: dict) -> str:
        """Create a standardized prompt for the model."""
        return f"""Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.

{example['header']}

{example['formal_theorem']} := by
  sorry"""

    def run_inference_parallel(self, prompts: list, max_steps: int = 1000) -> dict:
        """Run inference in parallel on a BATCH of prompts."""
        print(f"[Process {jax.process_index()}] Starting parallel inference...")
        start_time = time.time()
        
        try:
            num_devices = jax.local_device_count()
            prompt_batch = jnp.array(prompts).reshape((num_devices, -1))

            result = self.pmapped_sampler(
                prompt_batch,
                total_generation_steps=max_steps
            )
            result.block_until_ready()
            inference_time = time.time() - start_time
            
            generated_texts = result.text.flatten().tolist()
            
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

    def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
        """Verification logic. Runs only on process 0."""
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
        """Run the complete distributed test suite."""
        # This block now only runs on process 0.
        if jax.process_index() == 0:
            print("\n" + "=" * 80)
            print("Starting Herald Proofs DISTRIBUTED Inference & Verification")
            print("=" * 80)

            examples = self.load_herald_examples(num_examples=jax.device_count())
            if not examples:
                print("No examples loaded. Exiting.")
                return

            prompts = [self.create_prompt(ex) for ex in examples]
            
            # Since prompts only exist on process 0, we pass them to the parallel
            # function, and pmap handles the distribution to all devices.
            inference_result = self.run_inference_parallel(prompts)
            
            # Verification and summarization happens after results are gathered.
            print(f"\nParallel inference completed in {inference_result['inference_time']:.2f}s")
            
            results_data = []
            if inference_result['success']:
                for i, generated_text in enumerate(inference_result['generated_texts']):
                    example = examples[i]
                    print(f"\n--- Verifying EXAMPLE {i+1}/{len(examples)}: {example['name']} ---")
                    verification_result = self.verify_with_lean_compiler(generated_text, example['name'])
                    results_data.append({
                        'example': example,
                        'verified': verification_result['verified'],
                    })
            else:
                print(f"Inference failed: {inference_result['error']}")

            self._print_summary(results_data)

    def _print_summary(self, results: list):
        """Print a final summary (only on process 0)."""
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)

        verified_runs = [r for r in results if r.get('verified')]
        print(f"Total examples tested: {len(results)}")
        print(f"Successfully generated and verified proofs: {len(verified_runs)}/{len(results)}")
        print("=" * 80)

# def main():
#     """Main execution function for the script."""
#     # All processes initialize the class to load the model onto their devices.
#     tester = HeraldInferenceTester()
    
#     # Only process 0 orchestrates the test suite.
#     if jax.process_index() == 0:
#         tester.run_test_suite()

#     # Barrier to ensure all processes finish before exiting.
#     # This prevents process 0 from finishing while others might still be working.
#     jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count()))
    
#     if jax.process_index() == 0:
#         print("\nTest suite completed!")

# if __name__ == "__main__":
#     exit(main())


def main():
    """Main execution function for the script."""
    try:
        # Use the context manager to wrap the main workload.
        # This will automatically start and stop the profiler.
        with tpu_profiler.profile():
            tester = HeraldInferenceTester()
            tester.run_test_suite()

        # Barrier to ensure all processes finish before the script exits.
        jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count()))
        
        if jax.process_index() == 0:
            print("\nTest suite completed!")

    except Exception as e:
        print(f"\nA fatal error occurred in main on process {jax.process_index()}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
# #!/usr/bin/env python3
# """
# Lean Proofs Model Inference & Verification Script
# (Elegantly Refactored for Modularity and Reusability)
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

# class RecurrentGemmaService:
#     def __init__(self, ckpt_dir: Path, tok_file: Path):
#         print(f"[Process {jax.process_index()}] Initializing RecurrentGemma model...")
#         if not ckpt_dir.is_dir():
#             raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
#         if not tok_file.is_file():
#             raise FileNotFoundError(f"Tokenizer file not found: {tok_file}")

#         self._load_model_and_sampler(ckpt_dir, tok_file)
#         self._setup_pmap()
#         print(f"[Process {jax.process_index()}] Model and sampler loaded successfully!")

#     def _load_model_and_sampler(self, ckpt_dir: Path, tok_file: Path):
#         restored = ocp.PyTreeCheckpointer().restore(str(ckpt_dir))
#         params = restored.get("params", restored)
        
#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)
#         model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(tok_file))

#         self.sampler = rg.Sampler(
#             model=model,
#             params=jax_utils.replicate(params),
#             vocab=self.vocab
#         )

#     def _setup_pmap(self):
#         # FIX: The `static_argnums` argument is deprecated in newer JAX versions.
#         # Specifying `None` in `in_axes` is now the correct way to handle
#         # static, broadcasted arguments.
#         self.pmapped_generate = jax.pmap(
#             self.sampler,
#             in_axes=(0, None)
#         )

#     def generate(self, prompts: list[str], max_steps: int) -> list[str]:
#         """Runs parallel inference on a batch of prompts."""
#         tokenized_prompts = self.vocab.encode(prompts)
#         max_len = max(len(p) for p in tokenized_prompts)
#         padded_prompts = np.array(
#             [p + [self.vocab.pad_id()] * (max_len - len(p)) for p in tokenized_prompts]
#         )
        
#         num_devices = jax.local_device_count()
#         prompt_batch = padded_prompts.reshape((num_devices, -1, max_len))

#         result_tokens = self.pmapped_generate(prompt_batch, max_steps)
#         result_tokens.block_until_ready()
        
#         result_tokens_flat = result_tokens.reshape(-1, result_tokens.shape[-1])
#         return self.vocab.decode(result_tokens_flat.tolist())

# class LeanVerifier:
#     def __init__(self, lean_project_path: Path):
#         self.lean_project_path = lean_project_path
#         self.lean_src_path = self.lean_project_path / "LeanVerifier"
#         if not self.lean_src_path.is_dir():
#             raise FileNotFoundError(f"Lean source directory not found: {self.lean_src_path}")

#     def verify(self, lean_code: str, proof_name: str) -> dict:
#         """Verifies a string of Lean code by attempting to compile it."""
#         safe_filename = "".join(c if c.isalnum() else "_" for c in proof_name)
#         temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

#         try:
#             if ':= by' not in lean_code:
#                 return {'verified': False, 'output': "Separator ':= by' not found."}
#             if 'sorry' in lean_code.split(':= by', 1)[-1]:
#                 return {'verified': False, 'output': "Proof attempt used 'sorry'."}

#             temp_lean_file.write_text(lean_code, encoding='utf-8')
#             proc = subprocess.run(
#                 ['lake', 'build'],
#                 cwd=self.lean_project_path,
#                 capture_output=True, text=True, timeout=120
#             )
#             return {
#                 'verified': proc.returncode == 0,
#                 'output': proc.stdout if proc.returncode == 0 else proc.stderr
#             }
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
#         """Runs the complete test suite. Orchestrated by process 0."""
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
#             status = "✅ Verification successful!" if verification['verified'] else "❌ Verification failed."
#             print(status)
#             if not verification['verified']:
#                 print(f"   Reason: {verification['output'].strip().splitlines()[0]}")

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
Lean Proofs Model Inference & Verification Script
(Refactored for Multi-Worker TPU Execution – RecurrentGemma API compliant)
"""

import subprocess
import time
from pathlib import Path

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
        self.cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

        self.replicated_params = jax_utils.replicate(self.params)

        def generate_fn(params, tokenized_prompts, total_generation_steps):
            return rg.generate(
                prompt_tokens=tokenized_prompts,
                params=params,
                config=self.cfg,
                total_generation_steps=total_generation_steps,
            )

        self.pmapped_generate = jax.pmap(
            generate_fn,
            # Params are replicated, tokenized_prompts are sharded across devices.
            in_axes=(0, 0),
            static_broadcasted_argnums=(2,)  # total_generation_steps is static
        )

    def load_herald_examples(self, num_examples: int = 8):
        """Load examples. Only process 0 downloads the data."""
        if jax.process_index() != 0:
            return None  # Other processes don't load data.

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
            return None

    def create_prompt(self, example: dict) -> str:
        """Create a simple completion prompt suitable for a base model."""
        full_theorem = example.get('formal_theorem', '')
        try:
            proof_start_index = full_theorem.index(':= by')
            return full_theorem[:proof_start_index].strip()
        except ValueError:
            return full_theorem

    def run_inference_parallel(self, prompts: list, max_steps: int = 1000) -> dict:
        """Tokenizes prompts on CPU, then runs inference in parallel on TPUs."""
        print(f"[Process 0] Starting parallel inference...")
        start_time = time.time()
        
        try:
            # 1. Tokenize all prompts on the host CPU (process 0) using self.vocab
            tokenized_prompts = [self.vocab.encode(prompt) for prompt in prompts]

            # 2. Pad to the same length to create a single NumPy array
            max_len = max(len(p) for p in tokenized_prompts)
            padded_prompts = np.array(
                [p + [self.vocab.pad_id()] * (max_len - len(p)) for p in tokenized_prompts],
                dtype=np.int32
            )
            
            # 3. Reshape for pmap (num_devices, prompts_per_device, sequence_length)
            num_devices = jax.local_device_count()
            prompts_per_device = len(prompts) // num_devices
            prompt_batch = padded_prompts.reshape((num_devices, prompts_per_device, max_len))

            # 4. Run the pmapped generation function with the numerical data
            #    JAX will automatically handle sending the data from host 0 to all devices.
            result_tokens = self.pmapped_generate(
                self.replicated_params,
                prompt_batch,
                max_steps
            )
            result_tokens.block_until_ready()
            inference_time = time.time() - start_time
            
            # 5. Detokenize the results back to strings on the host CPU using self.vocab
            # The model output includes the prompt, so we decode the whole sequence.
            result_tokens_flat = result_tokens.reshape(-1, result_tokens.shape[-1])
            generated_texts = [self.vocab.decode(tokens) for tokens in result_tokens_flat.tolist()]
            
            return {
                'success': True,
                'generated_texts': generated_texts,  # This now contains the full code
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
                capture_output=True,
                text=True,
                timeout=120
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
        if jax.process_index() == 0:
            print("\n" + "=" * 80)
            print("Starting Herald Proofs DISTRIBUTED Inference & Verification")
            print("=" * 80)

            examples = self.load_herald_examples(num_examples=jax.device_count())
            if not examples:
                print("No examples loaded. Exiting.")
                return

            prompts = [self.create_prompt(ex) for ex in examples]
            
            inference_result = self.run_inference_parallel(prompts)
            
            print(f"\nParallel inference completed in {inference_result['inference_time']:.2f}s")
            
            results_data = []
            if inference_result['success']:
                # The generated_texts already contain the full theorem and proof.
                for i, full_generated_code in enumerate(inference_result['generated_texts']):
                    example = examples[i]
                    print(f"\n--- Verifying EXAMPLE {i+1}/{len(examples)}: {example['name']} ---")
                    # Directly verify the output from the model.
                    verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])
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

def main():
    """Main execution function for the script."""
    try:
        with tpu_profiler.profile():
            # All processes initialize the model and load it to their devices.
            tester = HeraldInferenceTester()
            # Only process 0 orchestrates the test suite.
            if jax.process_index() == 0:
                tester.run_test_suite()

        # Barrier to ensure all processes finish before exiting.
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
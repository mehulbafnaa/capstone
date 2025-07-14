#!/usr/bin/env python3
"""
Lean Proofs Model Inference & Verification Script
(Refactored for Multi-Worker TPU Execution & Configuration)
"""

import subprocess
import time
from pathlib import Path
import traceback
import argparse  # Import argparse

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

class HeraldInferenceTester:
    """
    Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
    """

    # REFACTOR: Accept arguments in __init__ instead of hardcoding paths.
    def __init__(self, ckpt_dir: Path, tok_file: Path, lean_project_path: Path):
        """Initialize the model and tokenizer for the current JAX process."""
        print(f"[Process {jax.process_index()}] Initializing RecurrentGemma model...")

        self.ckpt_dir = ckpt_dir
        self.tok_file = tok_file
        self.lean_project_path = lean_project_path
        self.lean_src_path = self.lean_project_path / "LeanVerifier"

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
        params = restored.get("params", restored)
        
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(params, preset=preset)
        model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))

        self.sampler = rg.Sampler(
            model=model,
            params=jax_utils.replicate(params),
            vocab=self.vocab
        )

        self.pmapped_generate = jax.pmap(
            self.sampler,
            in_axes=(0, None), 
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
            proof_start_index = full_theorem.index(':= by')
            return full_theorem[:proof_start_index] + ":="
        except ValueError:
            return full_theorem

    def run_inference_parallel(self, prompts: list, max_steps: int) -> dict:
        """Tokenize prompts on CPU, then run inference in parallel on TPUs."""
        print(f"[Process 0] Starting parallel inference with max_steps={max_steps}...")
        start_time = time.time()
        
        try:
            tokenized_prompts = self.vocab.encode(prompts)
            max_len = max(len(p) for p in tokenized_prompts)
            padded_prompts = np.array(
                [p + [self.vocab.pad_id()] * (max_len - len(p)) for p in tokenized_prompts]
            )
            
            num_devices = jax.local_device_count()
            prompt_batch = padded_prompts.reshape((num_devices, -1, max_len))

            result_tokens = self.pmapped_generate(
                prompt_batch,
                max_steps
            )
            result_tokens.block_until_ready()
            inference_time = time.time() - start_time
            
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

    def run_test_suite(self, num_examples: int, max_steps: int):
        """Run the complete distributed test suite. Orchestrated by process 0."""
        if jax.process_index() != 0:
            return

        print("\n" + "=" * 80)
        print("Starting Herald Proofs DISTRIBUTED Inference & Verification")
        print("=" * 80)
        
        examples = self.load_herald_examples(num_examples=num_examples)
        if not examples:
            print("No examples loaded. Exiting.")
            return

        prompts = [self.create_prompt(ex) for ex in examples]
        
        inference_result = self.run_inference_parallel(prompts, max_steps)
        
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
    # REFACTOR: Define and parse command-line arguments.
    parser = argparse.ArgumentParser(description="RecurrentGemma Lean Proof Inference & Verification")
    
    # Get the script's directory to set default paths
    script_dir = Path(__file__).parent.resolve()
    
    parser.add_argument("--ckpt_dir", type=Path, default=script_dir / "2b/2b",
                        help="Path to the model checkpoint directory.")
    parser.add_argument("--tok_file", type=Path, default=script_dir / "2b/tokenizer.model",
                        help="Path to the tokenizer model file.")
    parser.add_argument("--lean_project_path", type=Path, default=script_dir / "lean_verifier",
                        help="Path to the Lean verifier project.")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to test. Defaults to the number of available JAX devices.")
    parser.add_argument("--max_steps", type=int, default=1024,
                        help="Maximum generation steps for inference.")
    
    args = parser.parse_args()

    # If num_examples is not set, default it to the number of devices
    if args.num_examples is None:
        args.num_examples = jax.device_count()
        if args.num_examples == 0:
            print("⚠️ No JAX devices found. Defaulting to 1 example for CPU execution.")
            args.num_examples = 1

    try:
        # Pass parsed arguments to the tester
        tester = HeraldInferenceTester(
            ckpt_dir=args.ckpt_dir,
            tok_file=args.tok_file,
            lean_project_path=args.lean_project_path
        )
        
        jax.block_until_ready(jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count())))
        
        if jax.process_index() == 0:
            with tpu_profiler.profile():
                # Pass other arguments to the test suite
                tester.run_test_suite(num_examples=args.num_examples, max_steps=args.max_steps)
            print("\nTest suite completed!")

    except Exception as e:
        print(f"\n🚨 A fatal error occurred on process {jax.process_index()}: {e}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
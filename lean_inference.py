

#!/usr/bin/env python3
"""
Lean Proofs Model Inference & Verification Script

This script loads examples from the Herald Proofs dataset, runs inference
using a RecurrentGemma model, and then uses the Lean 4 compiler (via the
'lake' build tool) to formally verify the correctness of the generated proof.

It is designed to run from the root of the specific capstone project repository.
"""

import subprocess
import time
from pathlib import Path

import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset

# Get the absolute path of the directory containing this script (the repo root).
# This makes all other paths robust and independent of the current working directory.
SCRIPT_DIR = Path(__file__).parent.resolve()


class HeraldInferenceTester:
    """
    Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
    This class is configured for the specific repository structure provided.
    """

    def __init__(self):
        """Initialize the model and tokenizer using paths relative to the script."""
        print("Initializing RecurrentGemma model...")

        self.repo_root = SCRIPT_DIR
        self.ckpt_dir = self.repo_root / "2b-it" / "2b-it"
        self.tok_file = self.repo_root / "2b-it" / "tokenizer.model"
        self.lean_project_path = self.repo_root / "lean_verifier"

        # The source code directory inside the Lean project is "LeanVerifier" (capitalized).
        # We must point to it directly.
        self.lean_src_path = self.lean_project_path / "LeanVerifier"

        # Verify that necessary files and directories exist before proceeding.
        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found at: {self.ckpt_dir}")
        if not self.tok_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tok_file}")
        if not self.lean_src_path.exists():
            raise FileNotFoundError(f"Lean source directory not found at: {self.lean_src_path}")

        # Load the model and tokenizer.
        self._load_model()
        print("Model loaded successfully!")

    def _load_model(self):
        """Load the RecurrentGemma model, parameters, and tokenizer."""
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
            deterministic_sampling=True, # Use deterministic sampling for reproducible results.
            is_it_model=True
        )

    def load_herald_examples(self, num_examples: int = 3):
        """Load a specified number of random examples from the Herald Proofs dataset."""
        print(f"Loading {num_examples} examples from FrenzyMath/Herald_proofs dataset...")
        try:
            # Shuffle the dataset and take the first few examples for variety on each run.
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas().sample(frac=1).reset_index(drop=True)
            examples_data = df.head(num_examples)

            examples = []
            for i, row in examples_data.iterrows():
                example = row.to_dict()
                print(f"  Selected Example {i+1}: '{example['name']}'")
                examples.append(example)
            return examples
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def create_prompt(self, example: dict) -> str:
        """Create a standardized prompt for the model based on a dataset example."""
        prompt = f"""Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.

{example['header']}

{example['formal_theorem']} := by
  sorry"""
        return prompt

    def run_inference(self, prompt: str, max_steps: int = 1000) -> dict:
        """Run inference on a single prompt and time the operation."""
        print("Running inference...")
        start_time = time.time()
        try:
            result = self.sampler(
                [prompt],
                total_generation_steps=max_steps
            )
            inference_time = time.time() - start_time
            # The model is expected to return the full code, including the prompt.
            return {
                'success': True,
                'generated_text': result.text[0],
                'inference_time': inference_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    

    # def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
    #     """
    #     Writes the generated Lean code to a file and uses 'lake build' to verify it.
    #     """
    #     # Sanitize the example name to be a valid file name.
    #     safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
    #     temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

    #     try:
    #         # Write the generated code to a .lean file inside the project.
    #         temp_lean_file.write_text(full_code, encoding='utf-8')
    #         print(f"Verifying with Lean compiler by running 'lake build' in {self.lean_project_path}...")

    #         # Execute 'lake build' from within the project directory.
    #         proc = subprocess.run(
    #             ['lake', 'build'],
    #             cwd=self.lean_project_path,
    #             capture_output=True,
    #             text=True,
    #             timeout=120  # Add a 2-minute timeout to prevent hangs.
    #         )

    #         # Check the result of the compilation.
    #         if proc.returncode == 0:
    #             print("✅ Verification successful: Proof is correct!")
    #             return {'verified': True, 'output': proc.stdout}
    #         else:
    #             print("❌ Verification failed: Proof contains errors.")
    #             # The compiler error message is highly informative for debugging.
    #             return {'verified': False, 'output': proc.stderr}

    #     except subprocess.TimeoutExpired:
    #         print("❌ Verification timed out.")
    #         return {'verified': False, 'output': 'Compiler verification timed out.'}
    #     except Exception as e:
    #         print(f"An error occurred during verification: {e}")
    #         return {'verified': False, 'output': str(e)}
    #     finally:
    #         # Clean up by removing the temporary file.
    #         if temp_lean_file.exists():
    #             temp_lean_file.unlink()


    def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
        """
        Writes the generated Lean code to a file and uses 'lake build' to verify it.
        
        This version includes a critical check to ensure the proof does not use the
        'sorry' tactic, which would result in a false positive during compilation.
        """
        # --- Start of the refactored section ---

        # Step 1: Check for 'sorry' in the proof part of the generated code.
        # We split the code at ':= by' and check only the part that should be the proof.
        # This prevents false negatives if the word 'sorry' appears in a comment before the proof.
        proof_block = full_code.split(':= by', 1)[-1]
        if 'sorry' in proof_block:
            print("❌ Verification failed: Model used the 'sorry' tactic in its proof.")
            return {'verified': False, 'output': "Proof attempt used 'sorry'."}

        # --- End of the refactored section ---

        # Step 2: Proceed with compilation only if the proof is not using 'sorry'.
        
        # Sanitize the example name to be a valid file name.
        safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
        temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

        try:
            # Write the generated code to a .lean file inside the project.
            temp_lean_file.write_text(full_code, encoding='utf-8')
            print(f"Verifying with Lean compiler by running 'lake build' in {self.lean_project_path}...")

            # Execute 'lake build' from within the project directory.
            proc = subprocess.run(
                ['lake', 'build'],
                cwd=self.lean_project_path,
                capture_output=True,
                text=True,
                timeout=120  # Add a 2-minute timeout to prevent hangs.
            )

            # Check the result of the compilation.
            if proc.returncode == 0:
                print("✅ Verification successful: Proof is correct and does not use 'sorry'!")
                return {'verified': True, 'output': proc.stdout}
            else:
                print("❌ Verification failed: Proof contains compilation errors.")
                # The compiler error message is highly informative for debugging.
                return {'verified': False, 'output': proc.stderr}

        except subprocess.TimeoutExpired:
            print("❌ Verification timed out.")
            return {'verified': False, 'output': 'Compiler verification timed out.'}
        except Exception as e:
            print(f"An error occurred during verification: {e}")
            return {'verified': False, 'output': str(e)}
        finally:
            # Clean up by removing the temporary file.
            if temp_lean_file.exists():
                temp_lean_file.unlink()

    def run_test_suite(self):
        """Run the complete test suite: load, infer, verify, and report."""
        print("\n" + "=" * 80)
        print("Starting Herald Proofs Inference & Verification Test Suite")
        print("=" * 80)

        examples = self.load_herald_examples(3)
        if not examples:
            print("No examples loaded. Exiting.")
            return

        results = []
        for i, example in enumerate(examples, 1):
            print(f"\n--- EXAMPLE {i}/{len(examples)}: {example['name']} ---")

            prompt = self.create_prompt(example)
            inference_result = self.run_inference(prompt)

            if inference_result['success']:
                generated_text = inference_result['generated_text']
                print(f"Inference completed in {inference_result['inference_time']:.2f}s")

                # Verify the complete generated code with the Lean compiler.
                verification_result = self.verify_with_lean_compiler(generated_text, example['name'])

                print(f"\n--- Ground Truth Proof ---\n{example['formal_proof']}\n------------------------")
                print(f"\n--- Generated Full Output ---\n{generated_text}\n---------------------------")

                if not verification_result['verified']:
                    print(f"\n--- Compiler Errors ---\n{verification_result['output']}\n-----------------------")

                result_data = {
                    'example': example,
                    'generated_text': generated_text,
                    'verified': verification_result['verified'],
                    'compiler_output': verification_result['output'],
                    'inference_time': inference_result['inference_time']
                }
            else:
                print(f"Inference failed: {inference_result['error']}")
                result_data = {'example': example, 'error': inference_result['error']}

            results.append(result_data)
            print("-" * 60)

        self._print_summary(results)

    def _print_summary(self, results: list):
        """Print a final summary of all test results."""
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY (with Lean Compiler Verification)")
        print("=" * 80)

        successful_runs = [r for r in results if 'verified' in r]
        verified_runs = [r for r in successful_runs if r['verified']]

        print(f"Total examples tested: {len(results)}")
        print(f"Successfully generated and verified proofs: {len(verified_runs)}/{len(successful_runs)}")

        if successful_runs:
            avg_time = sum(r['inference_time'] for r in successful_runs) / len(successful_runs)
            print(f"Average inference time: {avg_time:.2f}s")

        print("\nIndividual Results:")
        for i, result in enumerate(results, 1):
            example_name = result['example']['name']
            if 'verified' in result:
                status = "✅ VERIFIED" if result['verified'] else "❌ FAILED"
                print(f"  {i}. {example_name}: {status}")
            else:
                print(f"  {i}. {example_name}: INFERENCE_ERROR")

        print("=" * 80)


def main():
    """Main execution function for the script."""
    try:
        tester = HeraldInferenceTester()
        tester.run_test_suite()
        print("\nTest suite completed!")
    except Exception as e:
        print(f"\nA fatal error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
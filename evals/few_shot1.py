#!/usr/bin/env python3
"""
Lean Proofs Model Inference & Verification Script (with Tagged Few-Shot Prompting)

This script loads examples from the Herald Proofs dataset, constructs a structured
few-shot prompt using XML-like tags to guide the model, runs inference, and then
uses the Lean 4 compiler to formally verify the correctness of the generated proof.
"""

import subprocess
import time
from pathlib import Path

import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset

# Get the absolute path of the repository root by going up two levels
# from this script's location (evals/few_shot.py -> evals/ -> repo_root/).
REPO_ROOT = Path(__file__).parent.parent.resolve()


class HeraldInferenceTester:
    """
    Tests and verifies a RecurrentGemma model on Herald Proofs dataset examples.
    """

    def __init__(self):
        """Initialize the model and tokenizer using paths relative to the repo root."""
        print("Initializing RecurrentGemma model...")

        self.ckpt_dir = REPO_ROOT / "2b" / "2b"
        self.tok_file = REPO_ROOT / "2b" / "tokenizer.model"
        self.lean_project_path = REPO_ROOT / "lean_verifier"
        self.lean_src_path = self.lean_project_path / "LeanVerifier"

        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found at: {self.ckpt_dir}")
        if not self.tok_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {self.tok_file}")
        if not self.lean_src_path.exists():
            raise FileNotFoundError(f"Lean source directory not found at: {self.lean_src_path}")

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
            deterministic_sampling=True,
            is_it_model=True
        )

    def load_herald_examples(self, num_total_examples: int = 4):
        """Load a specified number of random examples from the Herald Proofs dataset."""
        print(f"Loading {num_total_examples} examples from FrenzyMath/Herald_proofs dataset...")
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            df = dataset.to_pandas().sample(n=num_total_examples, random_state=42).reset_index(drop=True)
            
            examples = []
            for i, row in df.iterrows():
                examples.append(row.to_dict())
            return examples
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    # <<< MODIFIED: This function now uses XML-like tags for better structure >>>
    def create_few_shot_prompt(self, target_example: dict, few_shot_examples: list) -> str:
        """
        Creates a structured few-shot prompt using tags to clearly separate
        examples from the target problem.
        """
        prompt_parts = [
            "Complete the following Lean 4 theorem proof. Provide only the Lean tactics.",
            "Do not use 'sorry' or 'admit'. Do not provide any explanations, comments, or surrounding text."
        ]

        # Add the high-quality examples, wrapped in <example> tags.
        for ex in few_shot_examples:
            prompt_parts.append("\n<example>")
            proof_content = ex['formal_proof'].strip() if ex['formal_proof'] and ex['formal_proof'].strip() else "rfl"
            prompt_parts.append(f"{ex['header']}\n\n{ex['formal_theorem']} := by\n  {proof_content}")
            prompt_parts.append("</example>")

        # Add the final target problem, wrapped in <problem> tags.
        prompt_parts.append("\n<problem>")
        prompt_parts.append(f"{target_example['header']}\n\n{target_example['formal_theorem']} := by")
        # We do not close the problem tag, as the model should complete it.
        
        return "\n".join(prompt_parts)

    def run_inference(self, prompt: str, max_steps: int = 1000) -> dict:
        """Run inference on a single prompt and time the operation."""
        print("Running inference on structured few-shot prompt...")
        start_time = time.time()
        try:
            result = self.sampler(
                [prompt],
                total_generation_steps=max_steps
            )
            inference_time = time.time() - start_time
            return {
                'success': True,
                'generated_tactics': result.text[0],
                'inference_time': inference_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }

    def verify_with_lean_compiler(self, full_code: str, example_name: str) -> dict:
        """
        Writes the generated Lean code to a file and uses 'lake build' to verify it.
        Includes a check to ensure the proof is not using forbidden 'cheating' tactics.
        """
        proof_block = full_code.split(':= by', 1)[-1]
        forbidden_keywords = ['sorry', 'admit', 'axiom ']

        if any(keyword in proof_block for keyword in forbidden_keywords):
            print(f"❌ Verification failed: Model used a forbidden keyword ('{next(k for k in forbidden_keywords if k in proof_block)}')")
            return {'verified': False, 'output': "Proof attempt used a forbidden keyword."}

        safe_filename = "".join(c if c.isalnum() else "_" for c in example_name)
        temp_lean_file = self.lean_src_path / f"test_{safe_filename}.lean"

        try:
            temp_lean_file.write_text(full_code, encoding='utf-8')
            print(f"Verifying with Lean compiler by running 'lake build' in {self.lean_project_path}...")
            proc = subprocess.run(['lake', 'build'], cwd=self.lean_project_path, capture_output=True, text=True, timeout=120)

            if proc.returncode == 0:
                print("✅ Verification successful: Proof is correct and does not use 'sorry'!")
                return {'verified': True, 'output': proc.stdout}
            else:
                print("❌ Verification failed: Proof contains compilation errors.")
                return {'verified': False, 'output': proc.stderr}
        except subprocess.TimeoutExpired:
            print("❌ Verification timed out.")
            return {'verified': False, 'output': 'Compiler verification timed out.'}
        except Exception as e:
            print(f"An error occurred during verification: {e}")
            return {'verified': False, 'output': str(e)}
        finally:
            if temp_lean_file.exists():
                temp_lean_file.unlink()

    def run_test_suite(self, num_few_shot=1, num_to_test=3):
        """Run the complete test suite using a few-shot prompting strategy."""
        print("\n" + "=" * 80)
        print(f"Starting Herald Proofs Inference & Verification Test Suite ({num_few_shot}-Shot Strategy)")
        print("=" * 80)

        total_examples_to_load = num_few_shot + num_to_test
        all_examples = self.load_herald_examples(total_examples_to_load)
        if len(all_examples) < total_examples_to_load:
            print(f"Not enough examples loaded ({len(all_examples)}). Exiting.")
            return

        few_shot_examples = all_examples[:num_few_shot]
        test_examples = all_examples[num_few_shot:]
        
        print(f"Using {len(few_shot_examples)} example(s) for the few-shot prompt context.")
        print(f"Testing on {len(test_examples)} new examples.")

        results = []
        for i, example in enumerate(test_examples, 1):
            print(f"\n--- EXAMPLE {i}/{len(test_examples)}: {example['name']} ---")

            prompt = self.create_few_shot_prompt(example, few_shot_examples)
            
            inference_result = self.run_inference(prompt)

            if inference_result['success']:
                generated_tactics = inference_result['generated_tactics']
                # Clean up potential model artifacts like closing tags.
                if '</problem>' in generated_tactics:
                    generated_tactics = generated_tactics.split('</problem>')[0]
                
                print(f"Inference completed in {inference_result['inference_time']:.2f}s")
                
                full_generated_code = f"{example['header']}\n\n{example['formal_theorem']} := by\n  {generated_tactics}"

                verification_result = self.verify_with_lean_compiler(full_generated_code, example['name'])

                print(f"\n--- Ground Truth Proof ---\n{example['formal_proof']}\n------------------------")
                print(f"\n--- Generated Tactics ---\n{generated_tactics}\n---------------------------")

                if not verification_result['verified']:
                    print(f"\n--- Compiler Errors ---\n{verification_result['output']}\n-----------------------")

                result_data = {
                    'example': example,
                    'generated_tactics': generated_tactics,
                    'verified': verification_result['verified'],
                    'compiler_output': verification_result['output'],
                    'inference_time': inference_result['inference_time']
                }
            else:
                print(f"Inference failed: {inference_result['error']}")
                result_data = {'example': example, 'error': inference_result['error']}

            results.append(result_data)

        self._print_summary(results)

    def _print_summary(self, results: list):
        """Print a final summary of all test results."""
        print("\n" + "=" * 80)
        print(f"TEST SUITE SUMMARY ({len(results[0]['example'].get('few_shot_examples', [1]))}-Shot Verification with Tags)")
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
        # We will now run a "one-shot" test by default.
        tester.run_test_suite(num_few_shot=1, num_to_test=3)
        print("\nTest suite completed!")
    except Exception as e:
        print(f"\nA fatal error occurred in main: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
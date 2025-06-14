#!/usr/bin/env python3
"""
Herald Proofs Model Inference Test

This script loads 3 examples from the Herald Proofs dataset and runs inference
using the RecurrentGemma model to test performance on real mathematical proofs.
"""

from pathlib import Path
from datasets import load_dataset
import sentencepiece as spm
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import jax
import time


class HeraldInferenceTester:
    """
    Test RecurrentGemma model on Herald Proofs dataset examples
    """
    
    def __init__(self, ckpt_dir: str = "2b-it/2b-it", tok_file: str = "2b-it/tokenizer.model"):
        """Initialize the model and tokenizer"""
        print("Initializing RecurrentGemma model...")
        
        # File paths
        self.ckpt_dir = Path(ckpt_dir).resolve()
        self.tok_file = Path(tok_file).resolve()
        
        # Load model
        self._load_model()
        print("Model loaded successfully!")
        
    def _load_model(self):
        """Load the RecurrentGemma model and tokenizer"""
        # Restore weights
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        self.params = restored.get("params", restored)
        
        # Configure model
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        
        # Initialize components
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        self.sampler = rg.Sampler(
            model=self.model,
            vocab=self.vocab,
            params=self.params,
            deterministic_sampling=True,
            is_it_model=True
        )
    
    def load_herald_examples(self, num_examples: int = 3):
        """Load examples from Herald Proofs dataset"""
        print(f"Loading {num_examples} examples from Herald Proofs dataset...")
        
        try:
            dataset = load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True)
            
            # Convert to pandas for easier manipulation
            df = dataset.to_pandas()
            
            # Select diverse examples - let's pick some with different proof lengths
            df['formal_proof_len'] = df['formal_proof'].str.len()
            
            # Get examples: short, medium, and longer proof
            short_idx = df[df['formal_proof_len'] < 100].index[0] if len(df[df['formal_proof_len'] < 100]) > 0 else 0
            medium_idx = df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)].index[0] if len(df[(df['formal_proof_len'] >= 100) & (df['formal_proof_len'] < 300)]) > 0 else 1
            long_idx = df[df['formal_proof_len'] >= 300].index[0] if len(df[df['formal_proof_len'] >= 300]) > 0 else 2
            
            selected_indices = [short_idx, medium_idx, long_idx][:num_examples]
            examples = []
            
            for i, idx in enumerate(selected_indices):
                example = df.iloc[idx]
                examples.append({
                    'index': idx,
                    'id': example['id'],
                    'name': example['name'],
                    'header': example['header'],
                    'informal_theorem': example['informal_theorem'],
                    'formal_theorem': example['formal_theorem'],
                    'formal_proof': example['formal_proof'],
                    'informal_proof': example['informal_proof'],
                    'proof_length': len(example['formal_proof'])
                })
                print(f"  Example {i+1}: '{example['name']}' (proof length: {len(example['formal_proof'])} chars)")
            
            return examples
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def create_prompt(self, example):
        """Create a prompt for the model based on Herald dataset example"""
        
        prompt = f"""Complete the following Lean 4 theorem proof by replacing 'sorry' with the actual proof tactics.

{example['header']}

{example['formal_theorem']} := by
  sorry"""
        
        return prompt
    
    def run_inference(self, prompt: str, max_steps: int = 1000):
        """Run inference on a single prompt"""
        print("Running inference...")
        start_time = time.time()
        
        try:
            result = self.sampler(
                [prompt],
                total_generation_steps=max_steps
            )
            
            inference_time = time.time() - start_time
            generated_text = result.text[0]
            
            return {
                'success': True,
                'generated_text': generated_text,
                'inference_time': inference_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'inference_time': time.time() - start_time
            }
    
    def extract_proof_from_output(self, output_text: str, original_theorem: str):
        """Extract the proof portion from model output"""
        # The model should complete the theorem, so we need to extract everything after ":= by"
        lines = output_text.split('\n')
        
        # Find where the actual proof starts (after ":= by")
        proof_lines = []
        found_by = False
        
        for line in lines:
            if ':= by' in line and not found_by:
                found_by = True
                # Check if there's content after ":= by" on the same line
                by_index = line.find(':= by')
                after_by = line[by_index + 5:].strip()
                if after_by and after_by != 'sorry':
                    proof_lines.append(after_by)
                continue
            elif found_by and line.strip():
                # Skip lines that are just template remnants
                if 'sorry' not in line.strip() and '[your proof here]' not in line.strip():
                    proof_lines.append(line.strip())
        
        # If we found proof lines, return them
        if proof_lines:
            return '\n  '.join(proof_lines)
        
        # Fallback: look for the complete theorem in output
        if ':= by' in output_text:
            # Extract everything after the first ":= by"
            by_split = output_text.split(':= by', 1)
            if len(by_split) > 1:
                after_by = by_split[1].strip()
                # Clean up the proof part
                proof_lines = []
                for line in after_by.split('\n'):
                    line = line.strip()
                    if line and 'sorry' not in line and '[your proof here]' not in line:
                        proof_lines.append(line)
                if proof_lines:
                    return '\n  '.join(proof_lines)
        
        # If nothing found, return indication that no proof was generated
        return "No valid proof generated"
    
    def evaluate_example(self, example, generated_proof, ground_truth):
        """Simple evaluation of generated vs ground truth proof"""
        # Basic similarity checks
        generated_clean = generated_proof.replace(' ', '').replace('\n', '').lower()
        ground_truth_clean = ground_truth.replace(' ', '').replace('\n', '').lower()
        
        exact_match = generated_clean == ground_truth_clean
        
        # Check if key tactics are present
        ground_truth_tactics = set()
        for tactic in ['rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction']:
            if tactic in ground_truth.lower():
                ground_truth_tactics.add(tactic)
        
        generated_tactics = set()
        for tactic in ['rfl', 'simp', 'rw', 'exact', 'apply', 'intro', 'cases', 'induction']:
            if tactic in generated_proof.lower():
                generated_tactics.add(tactic)
        
        tactic_overlap = len(ground_truth_tactics.intersection(generated_tactics))
        tactic_total = len(ground_truth_tactics) if ground_truth_tactics else 1
        
        return {
            'exact_match': exact_match,
            'tactic_similarity': tactic_overlap / tactic_total,
            'ground_truth_tactics': ground_truth_tactics,
            'generated_tactics': generated_tactics
        }
    
    def run_test_suite(self):
        """Run the complete test suite on Herald examples"""
        print("Starting Herald Proofs inference test suite...")
        print("=" * 80)
        
        # Load examples
        examples = self.load_herald_examples(3)
        if not examples:
            print("No examples loaded. Exiting.")
            return
        
        results = []
        
        for i, example in enumerate(examples, 1):
            print(f"\nEXAMPLE {i}/3: {example['name']}")
            print("-" * 60)
            print(f"Theorem: {example['formal_theorem'][:100]}...")
            print(f"Ground truth proof length: {example['proof_length']} characters")
            
            # Create prompt
            prompt = self.create_prompt(example)
            
            # Run inference
            inference_result = self.run_inference(prompt)
            
            if inference_result['success']:
                generated_text = inference_result['generated_text']
                generated_proof = self.extract_proof_from_output(generated_text, example['formal_theorem'])
                
                # Evaluate
                evaluation = self.evaluate_example(
                    example, 
                    generated_proof, 
                    example['formal_proof']
                )
                
                print(f"Inference completed in {inference_result['inference_time']:.2f}s")
                print(f"Generated proof length: {len(generated_proof)} characters")
                print(f"Exact match: {evaluation['exact_match']}")
                print(f"Tactic similarity: {evaluation['tactic_similarity']:.2f}")
                
                print(f"\nFull Generated Output:")
                print("-" * 40)
                print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
                print("-" * 40)
                
                print(f"\nExtracted Proof:")
                print("-" * 40)
                print(generated_proof)
                print("-" * 40)
                
                print(f"\nGround Truth:")
                print("-" * 40)
                print(example['formal_proof'])
                print("-" * 40)
                
                result = {
                    'example': example,
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'generated_proof': generated_proof,
                    'evaluation': evaluation,
                    'inference_time': inference_result['inference_time']
                }
                
            else:
                print(f"Inference failed: {inference_result['error']}")
                result = {
                    'example': example,
                    'error': inference_result['error'],
                    'inference_time': inference_result['inference_time']
                }
            
            results.append(result)
            print("-" * 60)
        
        # Summary
        self._print_summary(results)
        return results
    
    def _print_summary(self, results):
        """Print summary of all test results"""
        print("\n" + "=" * 80)
        print("TEST SUITE SUMMARY")
        print("=" * 80)
        
        successful_runs = [r for r in results if 'generated_proof' in r]
        failed_runs = [r for r in results if 'error' in r]
        
        print(f"Successful inferences: {len(successful_runs)}/3")
        print(f"Failed inferences: {len(failed_runs)}/3")
        
        if successful_runs:
            avg_time = sum(r['inference_time'] for r in successful_runs) / len(successful_runs)
            exact_matches = sum(1 for r in successful_runs if r['evaluation']['exact_match'])
            avg_tactic_sim = sum(r['evaluation']['tactic_similarity'] for r in successful_runs) / len(successful_runs)
            
            print(f"Average inference time: {avg_time:.2f}s")
            print(f"Exact matches: {exact_matches}/{len(successful_runs)}")
            print(f"Average tactic similarity: {avg_tactic_sim:.2f}")
        
        print("\nIndividual Results:")
        for i, result in enumerate(results, 1):
            example_name = result['example']['name']
            if 'generated_proof' in result:
                match_status = "EXACT" if result['evaluation']['exact_match'] else "PARTIAL"
                print(f"  {i}. {example_name}: {match_status} (similarity: {result['evaluation']['tactic_similarity']:.2f})")
            else:
                print(f"  {i}. {example_name}: FAILED")
        
        print("=" * 80)


def main():
    """Main execution function"""
    try:
        # Initialize tester
        tester = HeraldInferenceTester()
        
        # Run test suite
        results = tester.run_test_suite()
        
        # Optionally save results
        print("\nSave detailed results to file? (y/n): ", end="")
        if input().lower().startswith('y'):
            import json
            with open('herald_inference_results.json', 'w') as f:
                # Convert results to JSON-serializable format
                json_results = []
                for r in results:
                    json_result = {
                        'example_name': r['example']['name'],
                        'example_id': r['example']['id'],
                        'inference_time': r.get('inference_time', 0),
                    }
                    if 'generated_proof' in r:
                        json_result.update({
                            'generated_proof': r['generated_proof'],
                            'exact_match': r['evaluation']['exact_match'],
                            'tactic_similarity': r['evaluation']['tactic_similarity']
                        })
                    else:
                        json_result['error'] = r.get('error', 'Unknown error')
                    json_results.append(json_result)
                
                json.dump(json_results, f, indent=2)
            print("Results saved to 'herald_inference_results.json'")
        
        print("\nTest suite completed!")
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
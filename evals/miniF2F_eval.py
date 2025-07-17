#!/usr/bin/env python3
"""
Unified High-Throughput Evaluation Script for miniF2F-lean4 Dataset
Adapted for miniF2F Olympiad-level mathematics problems with improved metrics
"""

import subprocess
import time
from pathlib import Path
import csv
from tqdm import tqdm
import warnings
import json
from multiprocessing import Pool, cpu_count
import argparse
import os
import uuid
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset

REPO_ROOT = Path(__file__).parent.parent.resolve()

@dataclass
class TacticAnalysis:
    valid_tactics: List[str]
    invalid_tactics: List[str]
    partial_progress: float
    syntax_score: float
    semantic_score: float
    forbidden_found: bool

@dataclass
class ProofResult:
    theorem_name: str
    verified: bool
    tactic_analysis: TacticAnalysis
    compilation_errors: List[str]
    partial_credit: float
    difficulty_level: str
    original_statement: str

def analyze_tactic_sequence(tactics: str, theorem_context: str, lean_project_path: Path) -> TacticAnalysis:
    """Analyze individual tactics for partial credit"""
    tactics_list = [t.strip() for t in tactics.split('\n') if t.strip()]
    valid_tactics = []
    invalid_tactics = []
    
    # Check for forbidden keywords first
    forbidden_keywords = ['sorry', 'admit', 'axiom ']
    proof_block = tactics
    forbidden_found = any(keyword in proof_block for keyword in forbidden_keywords)
    
    # Common valid Lean 4 tactics for mathematical reasoning
    lean4_tactics = {
        'rfl', 'simp', 'intro', 'intros', 'apply', 'exact', 'rw', 'rewrite',
        'cases', 'induction', 'constructor', 'left', 'right', 'exists',
        'use', 'have', 'show', 'calc', 'ring', 'field_simp', 'norm_num',
        'tauto', 'omega', 'linarith', 'norm_cast', 'abel', 'group',
        'rwa', 'simp_all', 'simp_rw', 'convert', 'congr', 'ext', 'funext',
        'push_neg', 'contrapose', 'by_contra', 'exfalso', 'trivial',
        'decide', 'norm_fin', 'interval_cases', 'fin_cases', 'mod_cases'
    }
    
    syntax_score = 0
    for tactic in tactics_list:
        if not tactic.strip():
            continue
            
        # Check if tactic starts with known Lean 4 keywords
        tactic_name = tactic.split()[0] if tactic.split() else ""
        if tactic_name in lean4_tactics:
            valid_tactics.append(tactic)
            syntax_score += 1
        elif _is_valid_lean4_syntax(tactic):
            valid_tactics.append(tactic)
            syntax_score += 0.5
        else:
            invalid_tactics.append(tactic)
    
    syntax_score = syntax_score / len(tactics_list) if tactics_list else 0
    
    # Estimate semantic correctness by attempting partial compilation
    semantic_score = _estimate_semantic_correctness(tactics, theorem_context, lean_project_path)
    
    return TacticAnalysis(
        valid_tactics=valid_tactics,
        invalid_tactics=invalid_tactics,
        partial_progress=len(valid_tactics) / len(tactics_list) if tactics_list else 0,
        syntax_score=syntax_score,
        semantic_score=semantic_score,
        forbidden_found=forbidden_found
    )

def _is_valid_lean4_syntax(tactic: str) -> bool:
    """Check if tactic follows Lean 4 syntax patterns"""
    patterns = [
        r'^\s*\w+\s*:=\s*.*',       # Variable assignment
        r'^\s*\w+\s+.*',            # Function application
        r'^\s*⟨.*⟩\s*$',            # Anonymous constructors
        r'^\s*\(.*\)\s*$',          # Parenthesized expressions
        r'^\s*\{.*\}\s*$',          # Set notation
        r'^\s*\[.*\]\s*$',          # List notation
        r'^\s*\w+\.\w+.*',          # Dot notation
        r'^\s*#\w+.*',              # Commands
    ]
    
    return any(re.match(pattern, tactic) for pattern in patterns)

def _estimate_semantic_correctness(tactics: str, theorem_context: str, lean_project_path: Path) -> float:
    """Estimate semantic correctness by progressive compilation"""
    if not tactics.strip():
        return 0.0
        
    lean_src_path = lean_project_path / "LeanVerifier"
    tactic_lines = [t.strip() for t in tactics.split('\n') if t.strip()]
    correct_prefixes = 0
    
    for i in range(1, len(tactic_lines) + 1):
        partial_tactics = '\n  '.join(tactic_lines[:i])
        test_code = f"{theorem_context} := by\n  {partial_tactics}\n  sorry"
        
        if _quick_compile_check(test_code, lean_project_path, lean_src_path):
            correct_prefixes = i
        else:
            break
    
    return correct_prefixes / len(tactic_lines) if tactic_lines else 0

def _quick_compile_check(code: str, lean_project_path: Path, lean_src_path: Path) -> bool:
    """Quick compilation check with timeout"""
    unique_id = str(uuid.uuid4())
    temp_file = lean_src_path / f"quick_check_{unique_id}.lean"
    
    try:
        temp_file.write_text(code, encoding='utf-8')
        proc = subprocess.run(
            ['lean', '--check', str(temp_file)], 
            cwd=lean_project_path,
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return proc.returncode == 0
    except:
        return False
    finally:
        if temp_file.exists():
            temp_file.unlink()

def _assess_difficulty(formal_statement: str) -> str:
    """Assess theorem difficulty based on mathematical content"""
    statement_lower = formal_statement.lower()
    
    # Advanced mathematical concepts
    advanced_keywords = ['continuous', 'differentiable', 'integral', 'derivative', 'limit', 
                         'topology', 'metric', 'convergence', 'series', 'infinite']
    
    # Intermediate concepts
    intermediate_keywords = ['∀', '∃', '→', '↔', 'induction', 'bijective', 'surjective', 
                            'injective', 'group', 'ring', 'field', 'prime', 'gcd']
    
    # Complex logical structure indicators
    complex_indicators = ['∀.*∃', '∃.*∀', '→.*→', '↔.*↔']
    
    if any(re.search(pattern, statement_lower) for pattern in complex_indicators):
        return "complex"
    elif any(word in statement_lower for word in advanced_keywords):
        return "advanced"
    elif any(word in statement_lower for word in intermediate_keywords):
        return "intermediate"
    else:
        return "basic"

def verify_single_proof_worker(record: dict) -> ProofResult:
    """
    Worker function to verify a single proof with comprehensive analysis
    """
    lean_project_path = REPO_ROOT / "lean_verifier"
    lean_src_path = lean_project_path / "LeanVerifier"

    # miniF2F-lean4 dataset structure
    formal_statement = record['formal_statement']
    generated_tactics = record['generated_tactics']
    theorem_name = record.get('name', record.get('id', 'unknown'))
    
    # Construct full code
    full_code = f"{formal_statement} := by\n  {generated_tactics}"
    
    # Analyze tactics comprehensively
    tactic_analysis = analyze_tactic_sequence(generated_tactics, formal_statement, lean_project_path)
    
    # Quick check for forbidden keywords
    if tactic_analysis.forbidden_found:
        return ProofResult(
            theorem_name=theorem_name,
            verified=False,
            tactic_analysis=tactic_analysis,
            compilation_errors=["forbidden_keyword"],
            partial_credit=0.0,
            difficulty_level=_assess_difficulty(formal_statement),
            original_statement=formal_statement
        )

    # Use UUID for guaranteed uniqueness
    unique_id = str(uuid.uuid4())
    temp_lean_file = lean_src_path / f"verify_{unique_id}.lean"

    try:
        temp_lean_file.write_text(full_code, encoding='utf-8')
        proc = subprocess.run(['lake', 'build'], cwd=lean_project_path, capture_output=True, text=True, timeout=120)
        
        if proc.returncode == 0:
            verified = True
            errors = []
        else:
            verified = False
            errors = _parse_compilation_errors(proc.stderr)
            
    except subprocess.TimeoutExpired:
        verified = False
        errors = ['timeout_error']
    except Exception as e:
        verified = False
        errors = [f'verification_exception: {str(e)}']
    finally:
        if temp_lean_file.exists():
            temp_lean_file.unlink()
    
    # Calculate partial credit
    partial_credit = _calculate_partial_credit(tactic_analysis, verified)
    
    return ProofResult(
        theorem_name=theorem_name,
        verified=verified,
        tactic_analysis=tactic_analysis,
        compilation_errors=errors,
        partial_credit=partial_credit,
        difficulty_level=_assess_difficulty(formal_statement),
        original_statement=formal_statement
    )

def _parse_compilation_errors(stderr: str) -> List[str]:
    """Parse and categorize compilation errors"""
    error_lines = [line.strip() for line in stderr.split('\n') if 'error:' in line.lower()]
    return error_lines[:5]  # Limit to first 5 errors

def _calculate_partial_credit(tactic_analysis: TacticAnalysis, verified: bool) -> float:
    """Calculate partial credit score"""
    if verified:
        return 1.0
    
    if tactic_analysis.forbidden_found:
        return 0.0
    
    # Weighted combination of different factors
    weights = {
        'syntax': 0.3,
        'semantics': 0.4,
        'progress': 0.3
    }
    
    return (
        weights['syntax'] * tactic_analysis.syntax_score +
        weights['semantics'] * tactic_analysis.semantic_score +
        weights['progress'] * tactic_analysis.partial_progress
    )

class UnifiedMiniF2FTester:
    def __init__(self):
        print(f"Host {jax.process_index()}: Initializing model...")
        self.ckpt_dir = REPO_ROOT / "2b" / "2b"
        self.tok_file = REPO_ROOT / "2b" / "tokenizer.model"
        for p in [self.ckpt_dir, self.tok_file]:
            if not p.exists():
                raise FileNotFoundError(f"Required path not found: {p}")
        self._load_model()
        print(f"Host {jax.process_index()}: Model loaded.")

    def _load_model(self):
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        self.params = restored.get("params", restored)
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=self.params, deterministic_sampling=True, is_it_model=False)

    def create_minif2f_prompt(self, target_example: dict, few_shot_examples: list) -> str:
        """Create specialized prompt for miniF2F mathematical problems"""
        prompt_parts = [
            "You are a Lean 4 theorem prover. Complete the following mathematical theorem proof.",
            "Provide only the Lean 4 tactics needed to prove the theorem.",
            "Do not use 'sorry', 'admit', or 'axiom'. Focus on rigorous mathematical reasoning.",
            "Use common Lean 4 tactics like: rfl, simp, intro, apply, exact, rw, cases, induction, ring, linarith, norm_num."
        ]
        
        # Add few-shot examples
        for i, ex in enumerate(few_shot_examples):
            prompt_parts.append(f"\n<example_{i+1}>")
            # Use the proof from the example if available
            proof_content = ex.get('proof', 'sorry')  # miniF2F might have ground truth proofs
            if proof_content and proof_content.strip() and proof_content.strip() != 'sorry':
                prompt_parts.append(f"{ex['formal_statement']} := by\n  {proof_content}")
            else:
                # If no proof available, show a template
                prompt_parts.append(f"{ex['formal_statement']} := by\n  sorry")
            prompt_parts.append(f"</example_{i+1}>")
        
        # Add the target problem
        prompt_parts.append("\n<problem>")
        prompt_parts.append(f"{target_example['formal_statement']} := by")
        
        return "\n".join(prompt_parts)

    def run_evaluation(self, num_few_shot: int, batch_size: int, is_dev_mode: bool, output_file: Path, split: str):
        """Run evaluation on miniF2F-lean4 dataset"""
        
        # Load miniF2F-lean4 dataset
        print(f"Host {jax.process_index()}: Loading miniF2F-lean4 dataset...")
        dataset = load_dataset("HaimingW/miniF2F-lean4", split=split, trust_remote_code=True)
        dataset = list(dataset)
        print(f"Host {jax.process_index()}: Loaded {len(dataset)} examples from {split} split")
        
        # Distribute the work across hosts
        few_shot_examples = dataset[:num_few_shot]
        full_test_dataset = dataset[num_few_shot:]
        my_test_slice = full_test_dataset[jax.process_index()::jax.process_count()]

        if is_dev_mode:
            my_test_slice = my_test_slice[:(batch_size * 2)]
            print(f"DEV MODE - Host {jax.process_index()}: Running on {len(my_test_slice)} examples.")

        # Phase 1: Inference
        print(f"\n--- Host {jax.process_index()}: Starting Phase 1: Inference ---")
        batches = [my_test_slice[i:i + batch_size] for i in range(0, len(my_test_slice), batch_size)]
        my_generated_results = []
        
        pbar_desc = f"Host {jax.process_index()} Inference"
        with tqdm(total=len(batches), desc=pbar_desc, position=jax.process_index()) as pbar:
            for batch in batches:
                prompts = [self.create_minif2f_prompt(ex, few_shot_examples) for ex in batch]
                inference_results = self.sampler(prompts, total_generation_steps=1024)

                for i, example in enumerate(batch):
                    generated_tactics = inference_results.text[i].split('</problem>')[0].strip()
                    my_generated_results.append({
                        "id": example.get("id", f"minif2f_{i}"),
                        "name": example.get("name", example.get("id", f"problem_{i}")),
                        "formal_statement": example["formal_statement"],
                        "generated_tactics": generated_tactics,
                        "ground_truth": example.get("proof", "")
                    })
                pbar.update(1)
        print(f"--- Host {jax.process_index()}: Phase 1: Inference Complete ---")

        # Phase 2: Verification with comprehensive analysis
        print(f"\n--- Host {jax.process_index()}: Starting Phase 2: Verification ---")
        
        # Setup CSV logging
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "theorem_name", "id", "verified", "partial_credit", "syntax_score", 
                "semantic_score", "difficulty", "valid_tactics", "invalid_tactics", 
                "compilation_errors", "forbidden_found", "generated_tactics", "ground_truth"
            ])
        
        num_processes = min(cpu_count(), len(my_generated_results))
        print(f"Host {jax.process_index()}: Verifying {len(my_generated_results)} proofs using {num_processes} local CPU cores...")

        with Pool(processes=num_processes) as pool:
            pbar_desc = f"Host {jax.process_index()} Verifying"
            pbar = tqdm(pool.imap_unordered(verify_single_proof_worker, my_generated_results), 
                       total=len(my_generated_results), desc=pbar_desc, position=jax.process_index())
            
            results = []
            for result in pbar:
                results.append(result)
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result.theorem_name, result.theorem_name, result.verified,
                        result.partial_credit, result.tactic_analysis.syntax_score,
                        result.tactic_analysis.semantic_score, result.difficulty_level,
                        ';'.join(result.tactic_analysis.valid_tactics),
                        ';'.join(result.tactic_analysis.invalid_tactics),
                        ';'.join(result.compilation_errors),
                        result.tactic_analysis.forbidden_found,
                        result.tactic_analysis.valid_tactics,  # Only valid tactics for readability
                        my_generated_results[results.index(result)].get("ground_truth", "")
                    ])
        
        # Calculate and display comprehensive metrics
        self._display_comprehensive_metrics(results)
        
        print(f"--- Host {jax.process_index()}: Phase 2: Verification Complete ---")

    def _display_comprehensive_metrics(self, results: List[ProofResult]):
        """Display comprehensive evaluation metrics"""
        if not results:
            return
            
        total = len(results)
        verified_count = sum(1 for r in results if r.verified)
        pass_rate = verified_count / total
        
        # Partial credit metrics
        avg_partial_credit = sum(r.partial_credit for r in results) / total
        avg_syntax_score = sum(r.tactic_analysis.syntax_score for r in results) / total
        avg_semantic_score = sum(r.tactic_analysis.semantic_score for r in results) / total
        
        # Difficulty breakdown
        difficulty_breakdown = {}
        for difficulty in ['basic', 'intermediate', 'advanced', 'complex']:
            subset = [r for r in results if r.difficulty_level == difficulty]
            if subset:
                difficulty_breakdown[difficulty] = {
                    'count': len(subset),
                    'pass_rate': sum(1 for r in subset if r.verified) / len(subset),
                    'avg_partial_credit': sum(r.partial_credit for r in subset) / len(subset)
                }
        
        print(f"\n--- Host {jax.process_index()}: EVALUATION METRICS ---")
        print(f"Total examples: {total}")
        print(f"Pass rate: {pass_rate:.3f} ({verified_count}/{total})")
        print(f"Average partial credit: {avg_partial_credit:.3f}")
        print(f"Average syntax score: {avg_syntax_score:.3f}")
        print(f"Average semantic score: {avg_semantic_score:.3f}")
        
        print(f"\nDifficulty breakdown:")
        for difficulty, stats in difficulty_breakdown.items():
            print(f"  {difficulty}: {stats['count']} examples, {stats['pass_rate']:.3f} pass rate, {stats['avg_partial_credit']:.3f} avg partial credit")

def main():
    parser = argparse.ArgumentParser(description="Run miniF2F-lean4 evaluation on single or multi-host TPUs.")
    parser.add_argument('--dev', action='store_true', help='Run in development mode.')
    parser.add_argument('--split', choices=['valid', 'test'], default='valid', help='Dataset split to use.')
    args = parser.parse_args()

    jax.distributed.initialize()

    print(f"Starting process {jax.process_index()} of {jax.process_count()} on host {jax.host_id()}.")
    
    if jax.process_index() == 0:
        print("="*80)
        mode = "DEVELOPMENT" if args.dev else "FULL EVALUATION"
        print(f"RUNNING miniF2F-lean4 EVALUATION IN **{mode} MODE**")
        print(f"Dataset split: {args.split}")
        print(f"JAX System: {jax.process_count()} processes, {jax.device_count()} total devices.")
        print("="*80)

    warnings.filterwarnings("ignore", message="Some donated buffers were not usable:")

    NUM_FEW_SHOT = 3  # miniF2F benefits from more examples
    BATCH_SIZE = 16 if args.dev else 32  # Smaller batches for complex problems
    
    base_filename = f"minif2f_dev_{args.split}" if args.dev else f"minif2f_results_{args.split}_{NUM_FEW_SHOT}-shot"
    output_filename = f"{base_filename}_host_{jax.process_index()}.csv"
    output_file = REPO_ROOT / output_filename

    try:
        tester = UnifiedMiniF2FTester()
        tester.run_evaluation(
            num_few_shot=NUM_FEW_SHOT,
            batch_size=BATCH_SIZE,
            is_dev_mode=args.dev,
            output_file=output_file,
            split=args.split
        )
        print(f"\nHost {jax.process_index()} evaluation complete!")
        print(f"Host {jax.process_index()} results logged to: {output_file}")
    except Exception as e:
        print(f"\nA fatal error occurred on host {jax.process_index()}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if jax.process_index() == 0:
        print("\n" + "="*80)
        print("miniF2F-lean4 evaluation complete on all hosts!")
        print("To combine results from all hosts:")
        print(f"  head -n 1 {output_file.name.replace(f'_host_{jax.process_index()}', '_host_0')}")
        print(f"  tail -n +2 -q {output_file.name.replace(f'_host_{jax.process_index()}', '_host_*')} >> minif2f_final_results.csv")
        print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())
# #!/usr/bin/env python3
# """
# Unified High-Throughput Evaluation Script for miniF2F-lean4 Dataset
# Adapted for miniF2F Olympiad-level mathematics problems with improved metrics
# Outputs a single merged JSON result file and suppresses the JAX fork warning.
# """

# # ------------------------------------------------------------------
# # 0.  Must be the first code executed – forces spawn-based multiprocessing
# # ------------------------------------------------------------------
# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)

# # ------------------------------------------------------------------
# # 1.  Standard imports
# # ------------------------------------------------------------------
# import subprocess
# import argparse
# import time
# import json
# import csv  # kept only for backward-compatibility; not used below
# from pathlib import Path
# from tqdm import tqdm
# import warnings
# import uuid
# import re
# from typing import List
# from dataclasses import dataclass, asdict

# import jax
# import jax.numpy as jnp  # only used by downstream utilities
# from jax.experimental import multihost_utils
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset

# REPO_ROOT = Path(__file__).parent.parent.resolve()

# # ------------------------------------------------------------------
# # 2.  Data classes
# # ------------------------------------------------------------------
# @dataclass
# class TacticAnalysis:
#     valid_tactics: List[str]
#     invalid_tactics: List[str]
#     partial_progress: float
#     syntax_score: float
#     semantic_score: float
#     forbidden_found: bool

# @dataclass
# class ProofResult:
#     theorem_name: str
#     verified: bool
#     tactic_analysis: TacticAnalysis
#     compilation_errors: List[str]
#     partial_credit: float
#     difficulty_level: str
#     original_statement: str

# # ------------------------------------------------------------------
# # 3.  All helper functions unchanged from your original file
# # ------------------------------------------------------------------
# def analyze_tactic_sequence(tactics: str, theorem_context: str, lean_project_path: Path) -> TacticAnalysis:
#     tactics_list = [t.strip() for t in tactics.split('\n') if t.strip()]
#     valid_tactics, invalid_tactics = [], []

#     forbidden_keywords = ['sorry', 'admit', 'axiom ']
#     forbidden_found = any(k in tactics for k in forbidden_keywords)

#     lean4_tactics = {
#         'rfl', 'simp', 'intro', 'intros', 'apply', 'exact', 'rw', 'rewrite',
#         'cases', 'induction', 'constructor', 'left', 'right', 'exists',
#         'use', 'have', 'show', 'calc', 'ring', 'field_simp', 'norm_num',
#         'tauto', 'omega', 'linarith', 'norm_cast', 'abel', 'group',
#         'rwa', 'simp_all', 'simp_rw', 'convert', 'congr', 'ext', 'funext',
#         'push_neg', 'contrapose', 'by_contra', 'exfalso', 'trivial',
#         'decide', 'norm_fin', 'interval_cases', 'fin_cases', 'mod_cases'
#     }

#     syntax_score = 0.0
#     for tactic in tactics_list:
#         if not tactic:
#             continue
#         tactic_name = tactic.split()[0] if tactic.split() else ""
#         if tactic_name in lean4_tactics:
#             valid_tactics.append(tactic)
#             syntax_score += 1.0
#         elif _is_valid_lean4_syntax(tactic):
#             valid_tactics.append(tactic)
#             syntax_score += 0.5
#         else:
#             invalid_tactics.append(tactic)

#     syntax_score = syntax_score / len(tactics_list) if tactics_list else 0.0
#     semantic_score = _estimate_semantic_correctness(tactics, theorem_context, lean_project_path)

#     return TacticAnalysis(
#         valid_tactics=valid_tactics,
#         invalid_tactics=invalid_tactics,
#         partial_progress=len(valid_tactics) / len(tactics_list) if tactics_list else 0.0,
#         syntax_score=syntax_score,
#         semantic_score=semantic_score,
#         forbidden_found=forbidden_found,
#     )

# def _is_valid_lean4_syntax(tactic: str) -> bool:
#     patterns = [
#         r'^\s*\w+\s*:=\s*.*',
#         r'^\s*\w+\s+.*',
#         r'^\s*⟨.*⟩\s*$',
#         r'^\s*\(.*\)\s*$',
#         r'^\s*\{.*\}\s*$',
#         r'^\s*\[.*\]\s*$',
#         r'^\s*\w+\.\w+.*',
#         r'^\s*#\w+.*',
#     ]
#     return any(re.match(p, tactic) for p in patterns)

# def _estimate_semantic_correctness(tactics: str, theorem_context: str, lean_project_path: Path) -> float:
#     if not tactics.strip():
#         return 0.0
#     tactic_lines = [t.strip() for t in tactics.split('\n') if t.strip()]
#     correct_prefixes = 0
#     for i in range(1, len(tactic_lines) + 1):
#         partial_tactics = '\n  '.join(tactic_lines[:i])
#         test_code = f"{theorem_context} := by\n  {partial_tactics}\n  sorry"
#         if _quick_compile_check(test_code, lean_project_path, lean_project_path / "LeanVerifier"):
#             correct_prefixes = i
#         else:
#             break
#     return correct_prefixes / len(tactic_lines) if tactic_lines else 0.0

# def _quick_compile_check(code: str, lean_project_path: Path, lean_src_path: Path) -> bool:
#     unique_id = str(uuid.uuid4())
#     temp_file = lean_src_path / f"quick_check_{unique_id}.lean"
#     try:
#         temp_file.write_text(code, encoding='utf-8')
#         proc = subprocess.run(
#             ['lean', '--check', str(temp_file)],
#             cwd=lean_project_path,
#             capture_output=True,
#             text=True,
#             timeout=10,
#         )
#         return proc.returncode == 0
#     except Exception:
#         return False
#     finally:
#         if temp_file.exists():
#             temp_file.unlink()

# def _assess_difficulty(formal_statement: str) -> str:
#     statement_lower = formal_statement.lower()
#     advanced_keywords = ['continuous', 'differentiable', 'integral', 'derivative', 'limit',
#                          'topology', 'metric', 'convergence', 'series', 'infinite']
#     intermediate_keywords = ['∀', '∃', '→', '↔', 'induction', 'bijective', 'surjective',
#                             'injective', 'group', 'ring', 'field', 'prime', 'gcd']
#     complex_indicators = ['∀.*∃', '∃.*∀', '→.*→', '↔.*↔']
#     if any(re.search(p, statement_lower) for p in complex_indicators):
#         return "complex"
#     elif any(k in statement_lower for k in advanced_keywords):
#         return "advanced"
#     elif any(k in statement_lower for k in intermediate_keywords):
#         return "intermediate"
#     return "basic"

# def verify_single_proof_worker(record: dict) -> ProofResult:
#     lean_project_path = REPO_ROOT / "lean_verifier"
#     lean_src_path = lean_project_path / "LeanVerifier"

#     formal_statement = record['formal_statement']
#     generated_tactics = record['generated_tactics']
#     theorem_name = record.get('name', record.get('id', 'unknown'))

#     full_code = f"{formal_statement} := by\n  {generated_tactics}"

#     tactic_analysis = analyze_tactic_sequence(generated_tactics, formal_statement, lean_project_path)
#     if tactic_analysis.forbidden_found:
#         return ProofResult(
#             theorem_name=theorem_name,
#             verified=False,
#             tactic_analysis=tactic_analysis,
#             compilation_errors=["forbidden_keyword"],
#             partial_credit=0.0,
#             difficulty_level=_assess_difficulty(formal_statement),
#             original_statement=formal_statement,
#         )

#     unique_id = str(uuid.uuid4())
#     temp_lean_file = lean_src_path / f"verify_{unique_id}.lean"
#     try:
#         temp_lean_file.write_text(full_code, encoding='utf-8')
#         proc = subprocess.run(
#             ['lake', 'build'],
#             cwd=lean_project_path,
#             capture_output=True,
#             text=True,
#             timeout=120,
#         )
#         verified = proc.returncode == 0
#         errors = [] if verified else [l.strip() for l in proc.stderr.split('\n') if 'error:' in l.lower()][:5]
#     except subprocess.TimeoutExpired:
#         verified, errors = False, ['timeout_error']
#     except Exception as e:
#         verified, errors = False, [f'verification_exception: {str(e)}']
#     finally:
#         if temp_lean_file.exists():
#             temp_lean_file.unlink()

#     partial_credit = 1.0 if verified else (
#         0.0 if tactic_analysis.forbidden_found else
#         0.3 * tactic_analysis.syntax_score +
#         0.4 * tactic_analysis.semantic_score +
#         0.3 * tactic_analysis.partial_progress
#     )

#     return ProofResult(
#         theorem_name=theorem_name,
#         verified=verified,
#         tactic_analysis=tactic_analysis,
#         compilation_errors=errors,
#         partial_credit=partial_credit,
#         difficulty_level=_assess_difficulty(formal_statement),
#         original_statement=formal_statement,
#     )

# # ------------------------------------------------------------------
# # 4.  UnifiedMiniF2FTester unchanged except JSON output & spawn Pool
# # ------------------------------------------------------------------
# class UnifiedMiniF2FTester:
#     def __init__(self):
#         print(f"Host {jax.process_index()}: Initializing model...")
#         self.ckpt_dir = REPO_ROOT / "2b" / "2b"
#         self.tok_file = REPO_ROOT / "2b" / "tokenizer.model"
#         if not self.ckpt_dir.exists() or not self.tok_file.exists():
#             raise FileNotFoundError("Required model / tokenizer not found")
#         restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(restored, preset=preset)
#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
#         self.sampler = rg.Sampler(
#             model=self.model,
#             vocab=self.vocab,
#             params=restored.get("params", restored),
#             deterministic_sampling=True,
#             is_it_model=False,
#         )

#     def create_minif2f_prompt(self, target_example: dict, few_shot_examples: list) -> str:
#         prompt_parts = [
#             "You are a Lean 4 theorem prover. Complete the following mathematical theorem proof.",
#             "Provide only the Lean 4 tactics needed to prove the theorem.",
#             "Do not use 'sorry', 'admit', or 'axiom'. Focus on rigorous mathematical reasoning.",
#             "Use common Lean 4 tactics like: rfl, simp, intro, apply, exact, rw, cases, induction, ring, linarith, norm_num.",
#         ]
#         for i, ex in enumerate(few_shot_examples):
#             prompt_parts.append(f"\n<example_{i+1}>")
#             proof = ex.get('proof', 'sorry')
#             prompt_parts.append(f"{ex['formal_statement']} := by\n  {proof}")
#             prompt_parts.append(f"</example_{i+1}>")
#         prompt_parts.append("\n<problem>")
#         prompt_parts.append(f"{target_example['formal_statement']} := by")
#         return "\n".join(prompt_parts)

#     def run_evaluation(self, num_few_shot: int, batch_size: int, is_dev_mode: bool, output_file: Path, split: str):
#         print(f"Host {jax.process_index()}: Loading miniF2F-lean4 dataset...")
#         dataset = list(load_dataset("HaimingW/miniF2F-lean4", split=split, trust_remote_code=True))
#         print(f"Host {jax.process_index()}: Loaded {len(dataset)} examples from {split} split")

#         few_shot_examples = dataset[:num_few_shot]
#         full_test_dataset = dataset[num_few_shot:]
#         my_test_slice = full_test_dataset[jax.process_index()::jax.process_count()]
#         if is_dev_mode:
#             my_test_slice = my_test_slice[:(batch_size * 2)]

#         # ------------------------------------------------------------------
#         # Phase 1: Inference
#         # ------------------------------------------------------------------
#         print(f"\n--- Host {jax.process_index()}: Starting Phase 1: Inference ---")
#         batches = [my_test_slice[i:i + batch_size] for i in range(0, len(my_test_slice), batch_size)]
#         my_results = []

#         pbar = tqdm(batches, desc=f"Host {jax.process_index()} Inference")
#         for batch in pbar:
#             prompts = [self.create_minif2f_prompt(ex, few_shot_examples) for ex in batch]
#             inference_results = self.sampler(prompts, total_generation_steps=1024)
#             for i, ex in enumerate(batch):
#                 generated = inference_results.text[i].split('</problem>')[0].strip()
#                 my_results.append({
#                     "id": ex.get("id", f"minif2f_{i}"),
#                     "name": ex.get("name", ex.get("id", f"problem_{i}")),
#                     "formal_statement": ex["formal_statement"],
#                     "generated_tactics": generated,
#                     "ground_truth": ex.get("proof", ""),
#                 })

#         # ------------------------------------------------------------------
#         # Phase 2: Verification (spawn-safe Pool)
#         # ------------------------------------------------------------------
#         print(f"\n--- Host {jax.process_index()}: Starting Phase 2: Verification ---")
#         num_processes = min(mp.cpu_count(), len(my_results))

#         # per-host JSON file
#         host_json = output_file.with_suffix('.json')
#         with open(host_json, 'w') as f:
#             json.dump([], f)

#         with mp.Pool(processes=num_processes, maxtasksperchild=1) as pool:
#             pbar = tqdm(pool.imap_unordered(verify_single_proof_worker, my_results),
#                         total=len(my_results), desc=f"Host {jax.process_index()} Verifying")
#             for res in pbar:
#                 with open(host_json, 'a') as f:
#                     json.dump(asdict(res), f)
#                     f.write('\n')

#         # ------------------------------------------------------------------
#         # Phase 3: Deterministic global merge (rank 0 only)
#         # ------------------------------------------------------------------
#         multihost_utils.sync_global_devices("eval_done")
#         # if jax.process_index() == 0:
#         #     combined = []
#         #     for host_id in range(jax.process_count()):
#         #         shard_path = output_file.with_suffix('.json').with_name(
#         #             f"{output_file.stem}_host_{host_id}.json"
#         #         )
#         #         with open(shard_path) as f:
#         #             for line in f:
#         #                 if line.strip():
#         #                     combined.append(json.loads(line))

#         #     final_path = output_file.with_suffix('.json').with_name(f"{output_file.stem}.json")
#         #     with open(final_path, 'w') as f:
#         #         json.dump(combined, f, indent=2)
#         #     print(f"Combined results saved to {final_path}")
#         if jax.process_index() == 0:
#             combined = []
#             for host_id in range(jax.process_count()):
#                 # build the correct shard file name
#                 shard_path = REPO_ROOT / f"minif2f_{split}_3-shot_host_{host_id}.json"
#                 with open(shard_path) as f:
#                     for line in f:
#                         if line.strip():
#                             combined.append(json.loads(line))

#             final_path = REPO_ROOT / f"minif2f_{split}_3-shot.json"
#             with open(final_path, 'w') as f:
#                 json.dump(combined, f, indent=2)
#             print(f"Combined results saved to {final_path}")

#         print(f"--- Host {jax.process_index()}: Phase 2: Verification Complete ---")

# # ------------------------------------------------------------------
# # 5.  Entry point
# # ------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Run miniF2F-lean4 evaluation on single or multi-host TPUs.")
#     parser.add_argument('--dev', action='store_true', help='Run in development mode.')
#     parser.add_argument('--split', choices=['valid', 'test'], default='valid', help='Dataset split to use.')
#     args = parser.parse_args()

#     jax.distributed.initialize()
#     if jax.process_index() == 0:
#         print("="*80)
#         print("RUNNING miniF2F-lean4 EVALUATION")
#         print(f"Mode: {'DEV' if args.dev else 'FULL'}")
#         print(f"Split: {args.split}")
#         print(f"Processes: {jax.process_count()}; Devices: {jax.device_count()}")
#         print("="*80)

#     NUM_FEW_SHOT = 3
#     BATCH_SIZE = 16 if args.dev else 32
#     base_filename = f"minif2f_{args.split}_3-shot"
#     output_file = REPO_ROOT / f"{base_filename}_host_{jax.process_index()}"

#     UnifiedMiniF2FTester().run_evaluation(
#         num_few_shot=NUM_FEW_SHOT,
#         batch_size=BATCH_SIZE,
#         is_dev_mode=args.dev,
#         output_file=output_file,
#         split=args.split,
#     )

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Unified High-Throughput Evaluation Script for miniF2F-lean4 Dataset
Outputs a single merged JSON file and suppresses JAX fork warnings.
"""

# 0.  MUST be first – ensure spawn-based multiprocessing
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

# ------------------------------------------------------------------
# 1.  Standard imports
# ------------------------------------------------------------------
import subprocess
import time
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse
import uuid
import re
from typing import List
from dataclasses import dataclass, asdict

import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset
from jax.experimental import multihost_utils

REPO_ROOT = Path(__file__).parent.parent.resolve()

# ------------------------------------------------------------------
# 2.  Data classes
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# 3.  Helper functions (unchanged)
# ------------------------------------------------------------------
def analyze_tactic_sequence(tactics: str, theorem_context: str, lean_project_path: Path) -> TacticAnalysis:
    tactics_list = [t.strip() for t in tactics.split('\n') if t.strip()]
    valid_tactics, invalid_tactics = [], []

    forbidden_keywords = ['sorry', 'admit', 'axiom ']
    forbidden_found = any(k in tactics for k in forbidden_keywords)

    lean4_tactics = {
        'rfl', 'simp', 'intro', 'intros', 'apply', 'exact', 'rw', 'rewrite',
        'cases', 'induction', 'constructor', 'left', 'right', 'exists',
        'use', 'have', 'show', 'calc', 'ring', 'field_simp', 'norm_num',
        'tauto', 'omega', 'linarith', 'norm_cast', 'abel', 'group',
        'rwa', 'simp_all', 'simp_rw', 'convert', 'congr', 'ext', 'funext',
        'push_neg', 'contrapose', 'by_contra', 'exfalso', 'trivial',
        'decide', 'norm_fin', 'interval_cases', 'fin_cases', 'mod_cases'
    }

    syntax_score = 0.0
    for tactic in tactics_list:
        if not tactic:
            continue
        tactic_name = tactic.split()[0] if tactic.split() else ""
        if tactic_name in lean4_tactics:
            valid_tactics.append(tactic)
            syntax_score += 1.0
        elif _is_valid_lean4_syntax(tactic):
            valid_tactics.append(tactic)
            syntax_score += 0.5
        else:
            invalid_tactics.append(tactic)

    syntax_score = syntax_score / len(tactics_list) if tactics_list else 0.0
    semantic_score = _estimate_semantic_correctness(tactics, theorem_context, lean_project_path)

    return TacticAnalysis(
        valid_tactics=valid_tactics,
        invalid_tactics=invalid_tactics,
        partial_progress=len(valid_tactics) / len(tactics_list) if tactics_list else 0.0,
        syntax_score=syntax_score,
        semantic_score=semantic_score,
        forbidden_found=forbidden_found,
    )

def _is_valid_lean4_syntax(tactic: str) -> bool:
    patterns = [
        r'^\s*\w+\s*:=\s*.*',
        r'^\s*\w+\s+.*',
        r'^\s*⟨.*⟩\s*$',
        r'^\s*\(.*\)\s*$',
        r'^\s*\{.*\}\s*$',
        r'^\s*\[.*\]\s*$',
        r'^\s*\w+\.\w+.*',
        r'^\s*#\w+.*',
    ]
    return any(re.match(p, tactic) for p in patterns)

def _estimate_semantic_correctness(tactics: str, theorem_context: str, lean_project_path: Path) -> float:
    if not tactics.strip():
        return 0.0
    tactic_lines = [t.strip() for t in tactics.split('\n') if t.strip()]
    correct_prefixes = 0
    for i in range(1, len(tactic_lines) + 1):
        partial_tactics = '\n  '.join(tactic_lines[:i])
        test_code = f"{theorem_context} := by\n  {partial_tactics}\n  sorry"
        if _quick_compile_check(test_code, lean_project_path, lean_project_path / "LeanVerifier"):
            correct_prefixes = i
        else:
            break
    return correct_prefixes / len(tactic_lines) if tactic_lines else 0.0

def _quick_compile_check(code: str, lean_project_path: Path, lean_src_path: Path) -> bool:
    unique_id = str(uuid.uuid4())
    temp_file = lean_src_path / f"quick_check_{unique_id}.lean"
    try:
        temp_file.write_text(code, encoding='utf-8')
        proc = subprocess.run(
            ['lean', '--check', str(temp_file)],
            cwd=lean_project_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False
    finally:
        if temp_file.exists():
            temp_file.unlink()

def _assess_difficulty(formal_statement: str) -> str:
    statement_lower = formal_statement.lower()
    advanced_keywords = ['continuous', 'differentiable', 'integral', 'derivative', 'limit',
                         'topology', 'metric', 'convergence', 'series', 'infinite']
    intermediate_keywords = ['∀', '∃', '→', '↔', 'induction', 'bijective', 'surjective',
                            'injective', 'group', 'ring', 'field', 'prime', 'gcd']
    complex_indicators = ['∀.*∃', '∃.*∀', '→.*→', '↔.*↔']
    if any(re.search(p, statement_lower) for p in complex_indicators):
        return "complex"
    elif any(k in statement_lower for k in advanced_keywords):
        return "advanced"
    elif any(k in statement_lower for k in intermediate_keywords):
        return "intermediate"
    return "basic"

def verify_single_proof_worker(record: dict) -> ProofResult:
    lean_project_path = REPO_ROOT / "lean_verifier"
    lean_src_path = lean_project_path / "LeanVerifier"

    formal_statement = record['formal_statement']
    generated_tactics = record['generated_tactics']
    theorem_name = record.get('name', record.get('id', 'unknown'))

    full_code = f"{formal_statement} := by\n  {generated_tactics}"

    tactic_analysis = analyze_tactic_sequence(generated_tactics, formal_statement, lean_project_path)
    if tactic_analysis.forbidden_found:
        return ProofResult(
            theorem_name=theorem_name,
            verified=False,
            tactic_analysis=tactic_analysis,
            compilation_errors=["forbidden_keyword"],
            partial_credit=0.0,
            difficulty_level=_assess_difficulty(formal_statement),
            original_statement=formal_statement,
        )

    unique_id = str(uuid.uuid4())
    temp_lean_file = lean_src_path / f"verify_{unique_id}.lean"
    try:
        temp_lean_file.write_text(full_code, encoding='utf-8')
        proc = subprocess.run(
            ['lake', 'build'],
            cwd=lean_project_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        verified = proc.returncode == 0
        errors = [] if verified else [l.strip() for l in proc.stderr.split('\n') if 'error:' in l.lower()][:5]
    except subprocess.TimeoutExpired:
        verified, errors = False, ['timeout_error']
    except Exception as e:
        verified, errors = False, [f'verification_exception: {str(e)}']
    finally:
        if temp_lean_file.exists():
            temp_lean_file.unlink()

    partial_credit = 1.0 if verified else (
        0.0 if tactic_analysis.forbidden_found else
        0.3 * tactic_analysis.syntax_score +
        0.4 * tactic_analysis.semantic_score +
        0.3 * tactic_analysis.partial_progress
    )

    return ProofResult(
        theorem_name=theorem_name,
        verified=verified,
        tactic_analysis=tactic_analysis,
        compilation_errors=errors,
        partial_credit=partial_credit,
        difficulty_level=_assess_difficulty(formal_statement),
        original_statement=formal_statement,
    )

# ------------------------------------------------------------------
# 4.  UnifiedMiniF2FTester
# ------------------------------------------------------------------
class UnifiedMiniF2FTester:
    def __init__(self):
        print(f"Host {jax.process_index()}: Initializing model...")
        self.ckpt_dir = REPO_ROOT / "2b" / "2b"
        self.tok_file = REPO_ROOT / "2b" / "tokenizer.model"
        if not self.ckpt_dir.exists() or not self.tok_file.exists():
            raise FileNotFoundError("Required model / tokenizer not found")
        restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
        preset = rg.Preset.RECURRENT_GEMMA_2B_V1
        cfg = rg.GriffinConfig.from_flax_params_or_variables(restored, preset=preset)
        self.model = rg.Griffin(cfg)
        self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
        self.sampler = rg.Sampler(
            model=self.model,
            vocab=self.vocab,
            params=restored.get("params", restored),
            deterministic_sampling=True,
            is_it_model=False,
        )

    def create_minif2f_prompt(self, target_example: dict, few_shot_examples: list) -> str:
        prompt_parts = [
            "You are a Lean 4 theorem prover. Complete the following mathematical theorem proof.",
            "Provide only the Lean 4 tactics needed to prove the theorem.",
            "Do not use 'sorry', 'admit', or 'axiom'. Focus on rigorous mathematical reasoning.",
            "Use common Lean 4 tactics like: rfl, simp, intro, apply, exact, rw, cases, induction, ring, linarith, norm_num.",
        ]
        for i, ex in enumerate(few_shot_examples):
            prompt_parts.append(f"\n<example_{i+1}>")
            proof = ex.get('proof', 'sorry')
            prompt_parts.append(f"{ex['formal_statement']} := by\n  {proof}")
            prompt_parts.append(f"</example_{i+1}>")
        prompt_parts.append("\n<problem>")
        prompt_parts.append(f"{target_example['formal_statement']} := by")
        return "\n".join(prompt_parts)

    def run_evaluation(self, num_few_shot: int, batch_size: int, is_dev_mode: bool, output_file: Path, split: str):
        print(f"Host {jax.process_index()}: Loading miniF2F-lean4 dataset...")
        dataset = list(load_dataset("HaimingW/miniF2F-lean4", split=split, trust_remote_code=True))
        print(f"Host {jax.process_index()}: Loaded {len(dataset)} examples from {split} split")

        few_shot_examples = dataset[:num_few_shot]
        full_test_dataset = dataset[num_few_shot:]
        my_test_slice = full_test_dataset[jax.process_index()::jax.process_count()]
        if is_dev_mode:
            my_test_slice = my_test_slice[:(batch_size * 2)]

        # ------------------------------------------------------------------
        # Phase 1: Inference
        # ------------------------------------------------------------------
        print(f"\n--- Host {jax.process_index()}: Starting Phase 1: Inference ---")
        batches = [my_test_slice[i:i + batch_size] for i in range(0, len(my_test_slice), batch_size)]
        my_results = []

        pbar = tqdm(batches, desc=f"Host {jax.process_index()} Inference")
        for batch in pbar:
            prompts = [self.create_minif2f_prompt(ex, few_shot_examples) for ex in batch]
            inference_results = self.sampler(prompts, total_generation_steps=1024)
            for i, ex in enumerate(batch):
                generated = inference_results.text[i].split('</problem>')[0].strip()
                my_results.append({
                    "id": ex.get("id", f"minif2f_{i}"),
                    "name": ex.get("name", ex.get("id", f"problem_{i}")),
                    "formal_statement": ex["formal_statement"],
                    "generated_tactics": generated,
                    "ground_truth": ex.get("proof", ""),
                })

        # ------------------------------------------------------------------
        # Phase 2: Verification (spawn-safe Pool + JSONL)
        # ------------------------------------------------------------------
        print(f"\n--- Host {jax.process_index()}: Starting Phase 2: Verification ---")
        num_processes = min(mp.cpu_count(), len(my_results))

        # per-host JSONL file (one object per line)
        host_jsonl = output_file.with_suffix('.jsonl')
        with open(host_jsonl, 'w') as f:
            pass  # create empty file

        with mp.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            pbar = tqdm(pool.imap_unordered(verify_single_proof_worker, my_results),
                        total=len(my_results), desc=f"Host {jax.process_index()} Verifying")
            for res in pbar:
                with open(host_jsonl, 'a') as f:
                    f.write(json.dumps(asdict(res)) + '\n')

        # ------------------------------------------------------------------
        # Phase 3: Deterministic global merge (rank 0 only)
        # ------------------------------------------------------------------
        multihost_utils.sync_global_devices("eval_done")
        if jax.process_index() == 0:
            combined = []
            for host_id in range(jax.process_count()):
                shard_path = REPO_ROOT / f"minif2f_{split}_3-shot_host_{host_id}.jsonl"
                with open(shard_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            combined.append(json.loads(line))

            final_path = REPO_ROOT / f"minif2f_{split}_3-shot.json"
            with open(final_path, 'w') as f:
                json.dump(combined, f, indent=2)
            print(f"Combined results saved to {final_path}")

        print(f"--- Host {jax.process_index()}: Phase 2: Verification Complete ---")

# ------------------------------------------------------------------
# 5.  Entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run miniF2F-lean4 evaluation on single or multi-host TPUs.")
    parser.add_argument('--dev', action='store_true', help='Run in development mode.')
    parser.add_argument('--split', choices=['valid', 'test'], default='valid', help='Dataset split to use.')
    args = parser.parse_args()

    jax.distributed.initialize()
    if jax.process_index() == 0:
        print("="*80)
        print("RUNNING miniF2F-lean4 EVALUATION")
        print(f"Mode: {'DEV' if args.dev else 'FULL'}")
        print(f"Split: {args.split}")
        print(f"Processes: {jax.process_count()}; Devices: {jax.device_count()}")
        print("="*80)

    NUM_FEW_SHOT = 3
    BATCH_SIZE = 16 if args.dev else 32
    split = args.split
    base_filename = f"minif2f_{split}_3-shot"
    output_file = REPO_ROOT / f"{base_filename}_host_{jax.process_index()}"

    UnifiedMiniF2FTester().run_evaluation(
        num_few_shot=NUM_FEW_SHOT,
        batch_size=BATCH_SIZE,
        is_dev_mode=args.dev,
        output_file=output_file,
        split=split,
    )

if __name__ == "__main__":
    main()
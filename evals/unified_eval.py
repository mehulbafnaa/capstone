# #!/usr/bin/env python3
# """
# Unified High-Throughput Evaluation Script (Batched & Parallelized)

# This single script performs a full-dataset evaluation in two phases:
# 1. A fast, batched inference phase that fully utilizes the distributed TPU.
# 2. A parallelized verification phase that uses all available CPU cores.
# This avoids the bottleneck of sequential verification without needing two scripts.
# """

# import subprocess
# import time
# from pathlib import Path
# import csv
# from tqdm import tqdm
# import warnings
# import json
# from multiprocessing import Pool, cpu_count
# import argparse

# import jax
# import orbax.checkpoint as ocp
# import recurrentgemma.jax as rg
# import sentencepiece as spm
# from datasets import load_dataset

# REPO_ROOT = Path(__file__).parent.parent.resolve()

# # This worker function is defined at the top level so multiprocessing can access it.
# def verify_single_proof_worker(record: dict) -> dict:
#     """
#     Worker function to verify a single proof. Runs in a separate process.
#     """
#     lean_project_path = REPO_ROOT / "lean_verifier"
#     lean_src_path = lean_project_path / "LeanVerifier"

#     full_code = f"{record['header']}\n\n{record['formal_theorem']} := by\n  {record['generated_tactics']}"
    
#     proof_block = full_code.split(':= by', 1)[-1]
#     forbidden_keywords = ['sorry', 'admit', 'axiom ']
#     if any(keyword in proof_block for keyword in forbidden_keywords):
#         record['verified'] = False
#         record['error_type'] = "forbidden_keyword"
#         return record

#     safe_filename = "".join(c if c.isalnum() else "_" for c in record['name'])
#     # Add process ID to filename to prevent collisions in multiprocessing
#     import os
#     import jax
#     temp_lean_file = lean_src_path / f"verify_{safe_filename}_{record['id']}_{os.getpid()}.lean"

#     try:
#         temp_lean_file.write_text(full_code, encoding='utf-8')
#         proc = subprocess.run(['lake', 'build'], cwd=lean_project_path, capture_output=True, text=True, timeout=120)
        
#         if proc.returncode == 0:
#             record['verified'] = True
#             record['error_type'] = None
#         else:
#             record['verified'] = False
#             record['error_type'] = "compilation_error"
#     except subprocess.TimeoutExpired:
#         record['verified'] = False
#         record['error_type'] = 'timeout_error'
#     except Exception:
#         record['verified'] = False
#         record['error_type'] = 'verification_exception'
#     finally:
#         if temp_lean_file.exists():
#             temp_lean_file.unlink()
    
#     return record


# class UnifiedTester:
#     def __init__(self):
#         print(f"Process {jax.process_index()}: Initializing model...")
#         self.ckpt_dir = REPO_ROOT / "2b" / "2b"
#         self.tok_file = REPO_ROOT / "2b" / "tokenizer.model"
#         for p in [self.ckpt_dir, self.tok_file]:
#             if not p.exists():
#                 raise FileNotFoundError(f"Required path not found: {p}")
#         self._load_model()
#         print(f"Process {jax.process_index()}: Model loaded.")

#     def _load_model(self):
#         restored = ocp.PyTreeCheckpointer().restore(str(self.ckpt_dir))
#         self.params = restored.get("params", restored)
#         preset = rg.Preset.RECURRENT_GEMMA_2B_V1
#         cfg = rg.GriffinConfig.from_flax_params_or_variables(self.params, preset=preset)
#         self.model = rg.Griffin(cfg)
#         self.vocab = spm.SentencePieceProcessor(model_file=str(self.tok_file))
#         self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=self.params, deterministic_sampling=True, is_it_model=True)

#     def create_few_shot_prompt(self, target_example: dict, few_shot_examples: list) -> str:
#         prompt_parts = [
#             "Complete the following Lean 4 theorem proof. Provide only the Lean tactics.",
#             "Do not use 'sorry' or 'admit'. Do not provide any explanations, comments, or surrounding text."
#         ]
#         for ex in few_shot_examples:
#             prompt_parts.append("\n<example>")
#             proof_content = ex['formal_proof'].strip() if ex['formal_proof'] and ex['formal_proof'].strip() else "rfl"
#             prompt_parts.append(f"{ex['header']}\n\n{ex['formal_theorem']} := by\n  {proof_content}")
#             prompt_parts.append("</example>")
#         prompt_parts.append("\n<problem>")
#         prompt_parts.append(f"{target_example['header']}\n\n{target_example['formal_theorem']} := by")
#         return "\n".join(prompt_parts)

#     def run_evaluation(self, num_few_shot: int, batch_size: int, is_dev_mode: bool, output_file: Path):
#         # Phase 1: Inference (TPU-focused)
#         # ==================================
#         # Only the main process loads data and drives the logic.
#         if jax.process_index() == 0:
#             print("\n--- Starting Phase 1: Inference ---")
#             dataset = list(load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True))
#             if is_dev_mode:
#                 dataset = dataset[:(batch_size * 2)]
#                 print(f"DEV MODE: Running inference on only {len(dataset)} examples.")
            
#             few_shot_examples = dataset[:num_few_shot]
#             test_dataset = dataset[num_few_shot:]
#             batches = [test_dataset[i:i + batch_size] for i in range(0, len(test_dataset), batch_size)]
            
#             all_generated_results = []
            
#             with tqdm(total=len(batches), desc="Phase 1: Inference", disable=(jax.process_index() != 0)) as pbar:
#                 for batch in batches:
#                     prompts = [self.create_few_shot_prompt(ex, few_shot_examples) for ex in batch]
#                     inference_results = self.sampler(prompts, total_generation_steps=1000)

#                     for i, example in enumerate(batch):
#                         generated_tactics = inference_results.text[i].split('</problem>')[0]
#                         all_generated_results.append({
#                             "id": example["id"], "name": example["name"], "header": example["header"],
#                             "formal_theorem": example["formal_theorem"], "generated_tactics": generated_tactics,
#                             "ground_truth": example["formal_proof"]
#                         })
#                     pbar.update(1)
#             print("--- Phase 1: Inference Complete ---")
#         else:
#             # Other processes do not need to do anything during this phase, but must be kept alive
#             # for the JAX collective calls inside the sampler. A more complex setup might use
#             # `jax.create_group` but for this architecture, letting them idle is simplest.
#             all_generated_results = None

#         # Phase 2: Verification (CPU-focused, only on main process)
#         # ========================================================
#         if jax.process_index() == 0:
#             print("\n--- Starting Phase 2: Verification ---")
            
#             # Setup CSV logging
#             with open(output_file, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["theorem_name", "id", "verified", "error_type", "generated_tactics", "ground_truth"])
            
#             num_processes = cpu_count()
#             print(f"Verifying {len(all_generated_results)} proofs using {num_processes} CPU cores...")

#             with Pool(processes=num_processes) as pool:
#                 pbar = tqdm(pool.imap_unordered(verify_single_proof_worker, all_generated_results), total=len(all_generated_results), desc="Phase 2: Verifying")
                
#                 for result in pbar:
#                     with open(output_file, 'a', newline='') as f:
#                         writer = csv.writer(f)
#                         writer.writerow([
#                             result.get("name"), result.get("id"), result.get("verified"),
#                             result.get("error_type"), result.get("generated_tactics"),
#                             result.get("ground_truth")
#                         ])
#             print("--- Phase 2: Verification Complete ---")

# def main():
#     parser = argparse.ArgumentParser(description="Run full or development evaluation.")
#     parser.add_argument('--dev', action='store_true', help='Run in development mode.')
#     args = parser.parse_args()

#     jax.distributed.initialize()

#     if jax.process_index() == 0:
#         print("="*80)
#         mode = "DEVELOPMENT" if args.dev else "FULL EVALUATION"
#         print(f"RUNNING IN **{mode} MODE**")
#         print(f"JAX System: {jax.process_count()} processes, {jax.device_count()} devices.")
#         print("="*80)

#     warnings.filterwarnings("ignore", message="Some donated buffers were not usable:")

#     NUM_FEW_SHOT = 2
#     BATCH_SIZE = 32 if args.dev else 128
    
#     output_filename = "dev_results.csv" if args.dev else f"unified_results_{NUM_FEW_SHOT}-shot.csv"
#     output_file = REPO_ROOT / output_filename

#     try:
#         tester = UnifiedTester()
#         tester.run_evaluation(
#             num_few_shot=NUM_FEW_SHOT,
#             batch_size=BATCH_SIZE,
#             is_dev_mode=args.dev,
#             output_file=output_file
#         )
#         if jax.process_index() == 0:
#             print("\nEvaluation complete!")
#             print(f"Final results logged to: {output_file}")
#     except Exception as e:
#         if jax.process_index() == 0:
#             print(f"\nA fatal error occurred: {e}")
#             import traceback
#             traceback.print_exc()
#         return 1
#     return 0

# if __name__ == "__main__":
#     exit(main())




#!/usr/bin/env python3
"""
Unified High-Throughput Evaluation Script (Batched & Parallelized)

This single script performs a full-dataset evaluation in two phases:
1. A fast, batched inference phase that fully utilizes the distributed TPU.
2. A parallelized verification phase that uses all available CPU cores on each host.
This avoids the bottleneck of sequential verification and scales to multi-host environments.
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

import jax
import orbax.checkpoint as ocp
import recurrentgemma.jax as rg
import sentencepiece as spm
from datasets import load_dataset

REPO_ROOT = Path(__file__).parent.parent.resolve()

# This worker function is defined at the top level so multiprocessing can access it.
def verify_single_proof_worker(record: dict) -> dict:
    """
    Worker function to verify a single proof. Runs in a separate process.
    """
    lean_project_path = REPO_ROOT / "lean_verifier"
    lean_src_path = lean_project_path / "LeanVerifier"

    full_code = f"{record['header']}\n\n{record['formal_theorem']} := by\n  {record['generated_tactics']}"
    
    proof_block = full_code.split(':= by', 1)[-1]
    forbidden_keywords = ['sorry', 'admit', 'axiom ']
    if any(keyword in proof_block for keyword in forbidden_keywords):
        record['verified'] = False
        record['error_type'] = "forbidden_keyword"
        return record

    safe_filename = "".join(c if c.isalnum() else "_" for c in record['name'])
    # Add JAX process index and OS process ID to filename for multi-host safety
    temp_lean_file = lean_src_path / f"verify_{safe_filename}_{record['id']}_{jax.process_index()}_{os.getpid()}.lean"

    try:
        temp_lean_file.write_text(full_code, encoding='utf-8')
        proc = subprocess.run(['lake', 'build'], cwd=lean_project_path, capture_output=True, text=True, timeout=120)
        
        if proc.returncode == 0:
            record['verified'] = True
            record['error_type'] = None
        else:
            record['verified'] = False
            # Capture more specific error info if possible
            record['error_type'] = "compilation_error"
            record['error_details'] = proc.stderr
    except subprocess.TimeoutExpired:
        record['verified'] = False
        record['error_type'] = 'timeout_error'
    except Exception as e:
        record['verified'] = False
        record['error_type'] = 'verification_exception'
        record['error_details'] = str(e)
    finally:
        if temp_lean_file.exists():
            temp_lean_file.unlink()
    
    return record


class UnifiedTester:
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
        self.sampler = rg.Sampler(model=self.model, vocab=self.vocab, params=self.params, deterministic_sampling=True, is_it_model=True)

    def create_few_shot_prompt(self, target_example: dict, few_shot_examples: list) -> str:
        prompt_parts = [
            "Complete the following Lean 4 theorem proof. Provide only the Lean tactics.",
            "Do not use 'sorry' or 'admit'. Do not provide any explanations, comments, or surrounding text."
        ]
        for ex in few_shot_examples:
            prompt_parts.append("\n<example>")
            proof_content = ex['formal_proof'].strip() if ex['formal_proof'] and ex['formal_proof'].strip() else "rfl"
            prompt_parts.append(f"{ex['header']}\n\n{ex['formal_theorem']} := by\n  {proof_content}")
            prompt_parts.append("</example>")
        prompt_parts.append("\n<problem>")
        prompt_parts.append(f"{target_example['header']}\n\n{target_example['formal_theorem']} := by")
        return "\n".join(prompt_parts)

    def run_evaluation(self, num_few_shot: int, batch_size: int, is_dev_mode: bool, output_file: Path):
        # This logic now runs on ALL hosts.
        
        # Each host loads the dataset info. Caching makes this fast.
        print(f"Host {jax.process_index()}: Loading dataset.")
        dataset = list(load_dataset("FrenzyMath/Herald_proofs", split="train", trust_remote_code=True))
        
        # Distribute the work across hosts.
        # Each host gets a unique, non-overlapping slice of the dataset.
        few_shot_examples = dataset[:num_few_shot]
        full_test_dataset = dataset[num_few_shot:]
        my_test_slice = full_test_dataset[jax.process_index()::jax.process_count()]

        if is_dev_mode:
            my_test_slice = my_test_slice[:(batch_size * 2)]
            print(f"DEV MODE - Host {jax.process_index()}: Running on {len(my_test_slice)} examples.")

        # Phase 1: Inference (Host-Parallel)
        # ==================================
        print(f"\n--- Host {jax.process_index()}: Starting Phase 1: Inference ---")
        batches = [my_test_slice[i:i + batch_size] for i in range(0, len(my_test_slice), batch_size)]
        my_generated_results = []
        
        # The progress bar is now specific to each host.
        pbar_desc = f"Host {jax.process_index()} Inference"
        with tqdm(total=len(batches), desc=pbar_desc, position=jax.process_index()) as pbar:
            for batch in batches:
                prompts = [self.create_few_shot_prompt(ex, few_shot_examples) for ex in batch]
                # The sampler uses the local TPUs attached to this host's process.
                inference_results = self.sampler(prompts, total_generation_steps=1000)

                for i, example in enumerate(batch):
                    # Robustly extract content before the first stop tag
                    generated_tactics = inference_results.text[i].split('</problem>')[0]
                    my_generated_results.append({
                        "id": example["id"], "name": example["name"], "header": example["header"],
                        "formal_theorem": example["formal_theorem"], "generated_tactics": generated_tactics,
                        "ground_truth": example["formal_proof"]
                    })
                pbar.update(1)
        print(f"--- Host {jax.process_index()}: Phase 1: Inference Complete ---")

        # Phase 2: Verification (Host-Parallel)
        # ========================================================
        print(f"\n--- Host {jax.process_index()}: Starting Phase 2: Verification ---")
        
        # Setup CSV logging for this host's output file
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["theorem_name", "id", "verified", "error_type", "generated_tactics", "ground_truth"])
        
        num_processes = cpu_count()
        print(f"Host {jax.process_index()}: Verifying {len(my_generated_results)} proofs using {num_processes} local CPU cores...")

        with Pool(processes=num_processes) as pool:
            pbar_desc = f"Host {jax.process_index()} Verifying"
            pbar = tqdm(pool.imap_unordered(verify_single_proof_worker, my_generated_results), total=len(my_generated_results), desc=pbar_desc, position=jax.process_index())
            
            for result in pbar:
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        result.get("name"), result.get("id"), result.get("verified"),
                        result.get("error_type"), result.get("generated_tactics"),
                        result.get("ground_truth")
                    ])
        print(f"--- Host {jax.process_index()}: Phase 2: Verification Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Run full or development evaluation on single or multi-host TPUs.")
    parser.add_argument('--dev', action='store_true', help='Run in development mode.')
    args = parser.parse_args()

    jax.distributed.initialize()

    # This print statement will now run on every host, which is informative.
    print(f"Starting process {jax.process_index()} of {jax.process_count()} on host {jax.host_id()}.")
    
    if jax.process_index() == 0:
        print("="*80)
        mode = "DEVELOPMENT" if args.dev else "FULL EVALUATION"
        print(f"RUNNING IN **{mode} MODE**")
        print(f"JAX System: {jax.process_count()} processes, {jax.device_count()} total devices.")
        print("="*80)

    warnings.filterwarnings("ignore", message="Some donated buffers were not usable:")

    NUM_FEW_SHOT = 2
    BATCH_SIZE = 32 if args.dev else 128
    
    # Generate a unique output file name for each host process
    base_filename = "dev_results" if args.dev else f"unified_results_{NUM_FEW_SHOT}-shot"
    output_filename = f"{base_filename}_host_{jax.process_index()}.csv"
    output_file = REPO_ROOT / output_filename

    try:
        tester = UnifiedTester()
        tester.run_evaluation(
            num_few_shot=NUM_FEW_SHOT,
            batch_size=BATCH_SIZE,
            is_dev_mode=args.dev,
            output_file=output_file
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
        print("All hosts finished. To combine the results, run the following commands:")
        print(f"  head -n 1 {output_file.name.replace(f'_host_{jax.process_index()}', '_host_0')}")
        print(f"  tail -n +2 -q {output_file.name.replace(f'_host_{jax.process_index()}', '_host_*')} >> final_results.csv")
        print("="*80)
    
    return 0

if __name__ == "__main__":
    exit(main())
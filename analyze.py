#!/usr/bin/env python3
"""
Quick utility to analyse token-length distribution of the pre-tokenised dataset.
Usage:
    python -m finetuning.analyze_seq_len
"""

from pathlib import Path
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
from finetuning.config import PRETOKENIZED_DATASET_DIR, TRAIN_SPLIT

def main():
    ds_path = Path(PRETOKENIZED_DATASET_DIR) / TRAIN_SPLIT
    if not ds_path.exists():
        raise FileNotFoundError(f"Pre-tokenised split not found at {ds_path}")

    ds = load_from_disk(str(ds_path))
    lengths = [len(ex["input_ids"]) for ex in ds]

    lengths = np.array(lengths)
    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    max_len = int(lengths.max())

    print("Sequence-length stats (tokens)")
    print(f"  Count          : {len(lengths)}")
    print(f"  Min            : {lengths.min()}")
    print(f"  Mean ± std     : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"  Median (50th)  : {int(np.percentile(lengths, 50))}")
    print(f"  95-th percentile: {p95}")
    print(f"  99-th percentile: {p99}")
    print(f"  Max             : {max_len}")
    print()

    # Conservative suggestion
    suggested = max(256, p95) if p95 < 512 else 512
    print(f"Recommended MAX_SEQ_LEN: {suggested}")

    # Optional quick plot
    try:
        plt.hist(lengths, bins=min(100, max_len // 8), edgecolor="black")
        plt.axvline(suggested, color="red", linestyle="--", label=f"Suggested ({suggested})")
        plt.xlabel("Sequence length (tokens)")
        plt.ylabel("Examples")
        plt.title("Token-length distribution")
        plt.legend()
        plt.tight_layout()
        out = Path(PRETOKENIZED_DATASET_DIR) / "seq_len_hist.png"
        plt.savefig(out)
        print(f"Histogram saved to {out}")
    except ImportError:
        print("matplotlib not found – skipping plot.")

if __name__ == "__main__":
    main()

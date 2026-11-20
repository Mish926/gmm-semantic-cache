"""Download only the columns we need from a vCache benchmark and save them locally.

Reads the class-label column, one embedding column and the prompt text straight from the
Hugging Face parquet file (HTTP range requests, no full download), parses the embedding
strings into a float32 matrix, and writes:

    data/processed/<name>.npz           X (n, d) float32, labels (n,) int32, ids (n,) int64
    data/processed/<name>_prompts.json  prompt text, same row order

Usage:
    python scripts/prepare_vcache.py                          # LM-Arena, gte embeddings
    python scripts/prepare_vcache.py --emb emb_e5_large_v2 --name lmarena_e5
    python scripts/prepare_vcache.py --limit 5000 --name lmarena_small   # quick test
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")  # the CDN timed out once at the default

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


def parse_embedding(text):
    """'[0.1, -0.2, ...]' -> float32 vector."""
    return np.fromstring(text.strip().lstrip("[").rstrip("]"), dtype=np.float32, sep=",")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="vCache/SemBenchmarkLmArena")
    p.add_argument("--file", default="train.parquet")
    p.add_argument("--emb", default="emb_gte", help="embedding column to keep")
    p.add_argument("--label-col", default="ID_Set", help="class column (ID_Set for LM-Arena, id_set for SearchQueries)")
    p.add_argument("--id-col", default="id")
    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--name", default="lmarena_gte")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--limit", type=int, default=None, help="only keep the first N rows")
    p.add_argument("--batch-size", type=int, default=2000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = f"datasets/{args.repo}/{args.file}"
    columns = [args.id_col, args.label_col, args.emb, args.prompt_col]
    print(f"reading {columns} from {path}")
    print("the embedding column is about 1 GB of text, so the first batch can take a few minutes...")

    fs = HfFileSystem()
    vectors, labels, ids, prompts = [], [], [], []
    skipped = 0
    t0 = time.time()
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        total = pf.metadata.num_rows if args.limit is None else min(args.limit, pf.metadata.num_rows)
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
            cols = batch.to_pydict()
            for rid, lab, emb, prompt in zip(cols[args.id_col], cols[args.label_col],
                                             cols[args.emb], cols[args.prompt_col]):
                if emb is None or lab is None:
                    skipped += 1
                    continue
                vectors.append(parse_embedding(emb))
                labels.append(str(lab))
                ids.append(int(rid))
                prompts.append(prompt or "")
                if len(vectors) >= total:
                    break
            print(f"  {len(vectors):,}/{total:,} rows  ({time.time() - t0:.0f}s)")
            if len(vectors) >= total:
                break

    dims = {v.shape[0] for v in vectors}
    if len(dims) != 1:
        raise ValueError(f"embeddings have inconsistent lengths: {sorted(dims)[:10]}")
    X = np.stack(vectors).astype(np.float32)
    classes, label_idx = np.unique(np.array(labels), return_inverse=True)

    np.savez_compressed(out_dir / f"{args.name}.npz",
                        X=X, labels=label_idx.astype(np.int32), ids=np.array(ids, dtype=np.int64),
                        class_names=classes.astype(str))
    with open(out_dir / f"{args.name}_prompts.json", "w") as f:
        json.dump(prompts, f)

    sizes = np.bincount(label_idx)
    print(f"\nsaved {out_dir / (args.name + '.npz')}")
    print(f"rows {X.shape[0]:,}   embedding dim {X.shape[1]}   classes {len(classes):,}   skipped {skipped}")
    print(f"prompts per class: min {sizes.min()}  median {int(np.median(sizes))}  max {sizes.max()}  "
          f"singletons {(sizes == 1).sum():,}")


if __name__ == "__main__":
    main()

"""Inspect the vCache semantic-caching benchmarks on Hugging Face without downloading them.

Prints, for each benchmark: the files and their sizes, the parquet schema (column names and
types), row counts, the length of any embedding columns, and a few example rows.
Only the parquet footer and one small slice are read, using HTTP range requests.

Usage:
    python scripts/inspect_vcache.py
    python scripts/inspect_vcache.py --repos vCache/SemBenchmarkLmArena
"""
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

DEFAULT_REPOS = [
    "vCache/SemBenchmarkLmArena",
    "vCache/SemBenchmarkClassification",
    "vCache/SemBenchmarkSearchQueries",
]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def short(value, width=100):
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[:width] + "..."


def inspect_parquet(fs, path, n_examples=3):
    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        schema = pf.schema_arrow
        meta = pf.metadata
        print(f"    rows: {meta.num_rows:,}   row groups: {meta.num_row_groups}")
        print("    columns:")
        for field in schema:
            print(f"      - {field.name}: {field.type}")

        scalar_cols = [f.name for f in schema
                       if not pa.types.is_list(f.type) and not pa.types.is_large_list(f.type)
                       and not pa.types.is_fixed_size_list(f.type)]
        list_cols = [f.name for f in schema if f.name not in scalar_cols]

        # vector columns: report their length from the first row only
        for col in list_cols:
            first = pf.read_row_group(0, columns=[col]).column(0)[0].as_py()
            length = len(first) if first is not None else None
            print(f"    vector column '{col}': length {length}")

        if scalar_cols:
            table = pf.read_row_group(0, columns=scalar_cols).slice(0, n_examples)
            print(f"    first {n_examples} rows (scalar columns):")
            for i, row in enumerate(table.to_pylist()):
                print(f"      [{i}]")
                for k, v in row.items():
                    print(f"         {k}: {short(v)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    args = p.parse_args()
    fs = HfFileSystem()

    for repo in args.repos:
        print("=" * 80)
        print(repo)
        print("=" * 80)
        root = f"datasets/{repo}"
        for entry in fs.find(root, detail=True).values():
            print(f"  {entry['name'].replace(root + '/', ''):60s} {human(entry.get('size', 0))}")
        parquets = sorted(fs.glob(f"{root}/**/*.parquet"))
        for path in parquets:
            print(f"\n  parquet: {path.replace(root + '/', '')}")
            try:
                inspect_parquet(fs, path)
            except Exception as exc:  # keep going if one file fails
                print(f"    could not read: {exc!r}")
        print()


if __name__ == "__main__":
    main()

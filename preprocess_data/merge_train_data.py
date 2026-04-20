"""Merge per-dataset SFT JSON files under sft_data/ into a single annotation JSON.

Recursively walks SFT_DATA_DIR, loads every JSON (each file is a list of
Qwen-VL SFT items), concatenates them, and writes the result to OUTPUT_PATH.
Items are kept as-is so `image` paths (relative to the processed_dir) still
resolve when the training config sets `data_path` to that directory.
"""

import argparse
import json
from pathlib import Path


DEFAULT_SFT_DATA_DIR = Path(
    "/home/user/research/samsung/mergeeda_v2/data/datasets/amba_document/sft_data"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/user/research/samsung/mergeeda_v2/data/datasets/amba_document/sft_data/merged_train.json"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SFT JSON files in sft_data/ into a single annotation file."
    )
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sft_dir.exists():
        raise FileNotFoundError(f"sft_data dir not found: {args.sft_dir}")

    output_path = args.output.resolve()
    json_files = sorted(
        p for p in args.sft_dir.rglob("*.json") if p.resolve() != output_path
    )
    if not json_files:
        print(f"No JSON files found under {args.sft_dir}")
        return

    merged: list[dict] = []
    per_file: list[tuple[Path, int]] = []
    for path in json_files:
        data = load_json(path)
        if not isinstance(data, list):
            print(f"Skipping non-list JSON: {path}")
            continue
        merged.extend(data)
        per_file.append((path, len(data)))

    save_json(output_path, merged)

    for path, count in per_file:
        print(f"{path.relative_to(args.sft_dir)}: {count}")
    print(f"\nTotal merged items: {len(merged)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import random
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/home/user/research/samsung/mergeeda_v2/data/datasets/amba_document/eval_qa"
)
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.7


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_items(items: list, train_ratio: float, seed: int) -> tuple[list, list]:
    shuffled_items = items.copy()
    random.Random(seed).shuffle(shuffled_items)

    train_size = int(len(shuffled_items) * train_ratio)
    train_items = shuffled_items[:train_size]
    test_items = shuffled_items[train_size:]
    return train_items, test_items


def is_split_file(path: Path) -> bool:
    return path.stem.endswith("_train") or path.stem.endswith("_test")


def split_json_file(path: Path, train_ratio: float, seed: int) -> tuple[Path, Path, int, int]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list.")

    train_data, test_data = split_items(data, train_ratio, seed)

    train_path = path.with_name(f"{path.stem}_train{path.suffix}")
    test_path = path.with_name(f"{path.stem}_test{path.suffix}")

    save_json(train_path, train_data)
    save_json(test_path, test_data)

    return train_path, test_path, len(train_data), len(test_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split AMBA eval QA JSON files into train/test JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing JSON files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Ratio for train split. Default: 0.7",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic shuffle. Default: 42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    json_paths = sorted(
        path
        for path in args.input_dir.rglob("*.json")
        if path.is_file() and not is_split_file(path)
    )

    if not json_paths:
        print(f"No JSON files found in {args.input_dir}")
        return

    for json_path in json_paths:
        train_path, test_path, train_count, test_count = split_json_file(
            json_path,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        print(
            f"{json_path} -> {train_path.name} ({train_count}), "
            f"{test_path.name} ({test_count})"
        )


if __name__ == "__main__":
    main()

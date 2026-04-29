"""SFT training data merge module for combining per-dataset JSON files."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainDataMerger:
    """Merge per-dataset SFT JSON files into a single annotation JSON.

    Recursively walks an sft_data directory, loads every JSON file (each
    expected to be a list of Qwen-VL SFT items), concatenates them, and
    writes the merged result to a specified output path. Items are kept
    as-is so image paths (relative to processed_dir) still resolve when
    the training config sets data_path to that directory.
    """

    def merge(
        self,
        sft_data_dir: str | Path,
        output_path: str | Path,
    ) -> None:
        """Merge all SFT JSON files under sft_data_dir into a single file.

        Skips the output file itself if it already exists inside sft_data_dir
        to avoid self-inclusion on re-runs.
        """
        sft_data_dir = Path(sft_data_dir)
        output_path = Path(output_path)

        if not sft_data_dir.exists():
            raise FileNotFoundError(f"sft_data dir not found: {sft_data_dir}")

        resolved_output = output_path.resolve()
        json_files = sorted(
            p for p in sft_data_dir.rglob("*.json") if p.resolve() != resolved_output
        )
        if not json_files:
            logger.warning(f"No JSON files found under {sft_data_dir}")
            return

        merged: list[dict] = []
        per_file: list[tuple[Path, int]] = []

        for path in json_files:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning(f"Skipping non-list JSON: {path}")
                continue
            merged.extend(data)
            per_file.append((path, len(data)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
            f.write("\n")

        for path, count in per_file:
            logger.info(f"{path.relative_to(sft_data_dir)}: {count} items")
        logger.info(f"Total merged items: {len(merged)} -> {output_path}")

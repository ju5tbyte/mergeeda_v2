"""Evaluation QA set generation module for all chunks in a dataset."""

import json
import logging
from pathlib import Path

from tqdm import tqdm

from .QGenerator import QGenerator

logger = logging.getLogger(__name__)


class EvalQSetGenerator:
    """Generate and save a typed QA evaluation set from a directory of text chunks."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
    ) -> None:
        """Initialize EvalQSetGenerator with an underlying QGenerator."""
        self._qa_generator = QGenerator(model=model, api_key=api_key)
        logger.info(f"EvalQSetGenerator initialized with model={model}")

    def generate(
        self,
        chunks_dir: str | Path,
        materials_dir: str | Path,
        output_path: str | Path,
        output_name: str,
    ) -> None:
        """Generate QA pairs for all chunks and save them split by question type.

        Iterates over all .md files in chunks_dir, calls QGenerator for each,
        collects results by type, and writes one JSON file per type:
            {output_path}/{output_name}_{type}.json
        """
        chunks_dir = Path(chunks_dir)
        materials_dir = Path(materials_dir)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        chunk_files = sorted(
            chunks_dir.glob("*.md"), key=lambda p: self._sort_key(p)
        )
        if not chunk_files:
            logger.warning(f"No .md files found in: {chunks_dir}")
            return

        logger.info(f"Processing {len(chunk_files)} chunks from: {chunks_dir}")

        questions_by_type: dict[str, list[dict]] = {}

        for chunk_file in tqdm(chunk_files, desc="Generating QA"):
            try:
                questions = self._qa_generator.generate(
                    chunk_path=chunk_file,
                    materials_dir=materials_dir,
                )
            except Exception as e:
                logger.error(
                    f"Failed to generate questions for {chunk_file.name}: {e}"
                )
                continue

            for q in questions:
                q_type = q.get("type", "unknown")
                questions_by_type.setdefault(q_type, []).append(q)

        logger.info(
            f"Total questions by type: { {t: len(qs) for t, qs in questions_by_type.items()} }"
        )

        self._save_by_type(questions_by_type, output_path, output_name)
        logger.info(f"QA sets saved to: {output_path}")

    def _save_by_type(
        self,
        questions_by_type: dict[str, list[dict]],
        output_path: Path,
        output_name: str,
    ) -> None:
        """Write one JSON file per question type."""
        for q_type, questions in questions_by_type.items():
            output_file = output_path / f"{output_name}_{q_type}.json"
            output_file.write_text(
                json.dumps(questions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                f"Saved {len(questions)} {q_type} questions -> {output_file}"
            )

    def _sort_key(self, path: Path) -> int:
        """Sort chunk files numerically by stem (e.g., '33.md' -> 33)."""
        try:
            return int(path.stem)
        except ValueError:
            return 0

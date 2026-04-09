"""Answer generation module using a Qwen VL model for evaluation QA sets."""

import base64
import json
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from mergeeda.models.builder import build_model
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generate model answers for evaluation QA sets and save unified preds.json.

    Reads all JSON QA files from an input directory, queries the Qwen VL model
    for each question (loading material images/tables where applicable), and
    writes a single preds.json to the output path.
    """

    def __init__(
        self,
        model_cfg: DictConfig,
    ) -> None:
        """Initialize AnswerGenerator by building the model from config."""
        self._model = build_model(model_cfg)
        logger.info(f"AnswerGenerator initialized with model: {model_cfg.name}")

    def generate(
        self,
        qa_dir: str | Path,
        materials_dir: str | Path,
        output_path: str | Path,
    ) -> None:
        """Generate answers for all QA JSON files in qa_dir and save preds.json.

        Iterates over all .json files in qa_dir sorted by filename, assigns
        sequential IDs across files (ascending filename order, then original
        order within each file), queries the model, and writes preds.json.
        """
        qa_dir = Path(qa_dir)
        materials_dir = Path(materials_dir)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        json_files = sorted(qa_dir.glob("*.json"), key=lambda p: p.name)
        if not json_files:
            logger.warning(f"No .json files found in: {qa_dir}")
            return

        logger.info(f"Processing {len(json_files)} QA files from: {qa_dir}")

        all_items: list[dict] = []
        for json_file in json_files:
            items = self._load_qa_file(json_file)
            all_items.extend(items)
            logger.info(f"Loaded {len(items)} questions from: {json_file.name}")

        results: list[dict] = []
        for idx, item in enumerate(tqdm(all_items, desc="Generating answers"), start=1):
            answer = self._query_model(item, materials_dir)
            result = {k: v for k, v in item.items()}
            result["id"] = idx
            result["answer"] = answer
            # reorder: id first
            ordered = {"id": result.pop("id")}
            ordered.update(result)
            ordered["answer"] = ordered.pop("answer")
            results.append(ordered)

        output_file = output_path / "preds.json"
        output_file.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved {len(results)} predictions -> {output_file}")

    def _load_qa_file(self, json_file: Path) -> list[dict]:
        """Load and return items from a single QA JSON file."""
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.warning(f"Unexpected format in {json_file.name}, skipping")
                return []
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {json_file.name}: {e}")
            return []

    def _query_model(self, item: dict, materials_dir: Path) -> str:
        """Query the model with a question and optional material images.

        For material-type questions, loads the referenced image file (if it is
        an image) to pass as visual context. Table (.txt) materials are
        appended as text to the question.
        """
        question = item.get("question", "")
        material_filename: str | None = item.get("material")

        imgs: list[Image.Image] = []
        extra_text = ""

        if material_filename:
            material_path = materials_dir / material_filename
            if not material_path.exists():
                logger.warning(f"Material file not found: {material_path}")
            else:
                suffix = material_path.suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                    try:
                        imgs.append(Image.open(material_path).convert("RGB"))
                    except OSError as e:
                        logger.warning(f"Failed to open image {material_path}: {e}")
                elif suffix == ".txt":
                    try:
                        table_text = material_path.read_text(encoding="utf-8")
                        extra_text = f"\n\n[Table: {material_filename}]\n{table_text}"
                    except OSError as e:
                        logger.warning(f"Failed to read table {material_path}: {e}")
                else:
                    logger.warning(f"Unsupported material type, skipping: {material_filename}")

        full_question = question + extra_text

        try:
            answer = self._model(full_question, imgs if imgs else None)
        except Exception as e:
            logger.error(f"Model inference failed for question: {question[:60]}...: {e}")
            answer = ""

        return answer

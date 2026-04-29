"""SFT training data generation module using GPT to answer train QA pairs."""

import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

MATERIAL_TAG_PATTERN = re.compile(r"<material:([\w\-._]+)>")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_SYSTEM_PROMPT = """You are an expert on the AMBA (Advanced Microcontroller Bus Architecture) specifications.

You are given:
- A text chunk extracted from an AMBA specification document.
- Optional materials referenced by the chunk: tables provided as labeled text blocks ([Table: FILENAME]) and figures provided as inline images, in the order they appear in the text.
- A question about the content of that chunk.

Write a single, high-quality answer to the question that will be used as a supervised fine-tuning target for a vision-language model.

Rules:
1. Ground the answer strictly in the provided text chunk and materials. Do not invent facts, signals, or behaviors that are not supported by what you are shown.
2. The answer must be SELF-CONTAINED and authoritative. Do NOT reference "the passage", "the chunk", "the provided text", "the figure above", section numbers, or the source in any form. Write as if you are stating the specification directly.
3. Be technically precise: use the exact signal names, field names, and terminology from the AMBA specification as they appear in the text / materials.
4. Keep the answer concise but complete. Cover every point needed to fully answer the question; do not add unrelated background.
5. For questions that reference a figure or table, interpret the relevant material to produce the answer, but still do not mention the figure/table identifier in your response.
6. Output ONLY the answer as plain text. No JSON, no preamble, no markdown code fences, no lists unless genuinely warranted by the content."""


class TrainDataGenerator:
    """Generate Qwen-VL SFT training data by answering QA pairs via GPT.

    Resolves each question's source chunk from the processed dataset directory,
    calls the GPT model with the chunk text and any referenced materials, and
    saves results in the Qwen-VL finetune schema consumed by data_processor.py.
    Supports incremental resume: already-generated items are skipped on restart.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
    ) -> None:
        """Initialize TrainDataGenerator with an OpenAI client."""
        self._model = model
        self._client = OpenAI(api_key=api_key)
        logger.info(f"TrainDataGenerator initialized with model={model}")

    def generate(
        self,
        train_file: str | Path,
        dataset_name: str,
        chunks_dir: str | Path,
        materials_dir: str | Path,
        output_file: str | Path,
        overwrite: bool = False,
    ) -> None:
        """Generate SFT items for a single train QA JSON file and save results.

        Reads the train_file (list of QA dicts), resolves each source chunk,
        calls GPT, and writes results incrementally to output_file. Skips items
        already present in output_file unless overwrite is True.
        """
        train_file = Path(train_file)
        chunks_dir = Path(chunks_dir)
        materials_dir = Path(materials_dir)
        output_file = Path(output_file)

        items = self._load_json(train_file)
        if not isinstance(items, list):
            logger.warning(f"Unexpected format (not a list): {train_file}")
            return

        if overwrite and output_file.exists():
            output_file.unlink()

        results: list[dict] = []
        if output_file.exists():
            existing = self._load_json(output_file)
            results = existing if isinstance(existing, list) else []

        done_keys: set[tuple[str, str]] = {
            (r.get("source_chunk", ""), self._human_text(r)) for r in results
        }
        chunk_cache: dict[str, str] = {}

        desc = f"{train_file.parent.name}/{train_file.name}"
        for item in tqdm(items, desc=desc, leave=False):
            question = item.get("question", "")
            source_chunk = item.get("source_chunk")
            if not question or not source_chunk:
                logger.warning(f"Skipping item missing question/source_chunk: {item}")
                continue

            resume_candidates = {
                (source_chunk, question),
                (source_chunk, f"{question}\n<image>"),
            }
            if done_keys & resume_candidates:
                continue

            if source_chunk not in chunk_cache:
                chunk_path = chunks_dir / source_chunk
                if not chunk_path.exists():
                    logger.warning(f"Chunk not found, skipping: {chunk_path}")
                    continue
                chunk_cache[source_chunk] = chunk_path.read_text(encoding="utf-8")
            chunk_text = chunk_cache[source_chunk]

            try:
                answer = self._call_gpt(question, chunk_text, materials_dir)
            except Exception as e:
                logger.error(f"GPT call failed for {source_chunk}: {e}")
                answer = ""

            sft_item = self._build_sft_item(item, answer, dataset_name, materials_dir)
            results.append(sft_item)
            done_keys.add((source_chunk, self._human_text(sft_item)))

            self._save_json(output_file, results)

        logger.info(f"Wrote {len(results)} items -> {output_file}")

    def _call_gpt(
        self,
        question: str,
        chunk_text: str,
        materials_dir: Path,
    ) -> str:
        """Call GPT with a chunk + question and return the answer text."""
        messages = self._build_messages(question, chunk_text, materials_dir)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    def _build_messages(
        self,
        question: str,
        chunk_text: str,
        materials_dir: Path,
    ) -> list[dict]:
        """Assemble the system + user message list for the GPT call."""
        material_filenames = MATERIAL_TAG_PATTERN.findall(chunk_text)
        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "<text_chunk>\n"
                    f"{chunk_text}\n"
                    "</text_chunk>\n\n"
                    f"Question: {question}\n\n"
                    "Write the answer."
                ),
            }
        ]
        user_content.extend(self._build_material_blocks(material_filenames, materials_dir))
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_material_blocks(
        self,
        material_filenames: list[str],
        materials_dir: Path,
    ) -> list[dict]:
        """Build OpenAI content blocks for each material referenced in a chunk."""
        blocks: list[dict] = []
        for filename in material_filenames:
            material_path = materials_dir / filename
            if not material_path.exists():
                logger.warning(f"Material not found, skipping: {material_path}")
                continue
            suffix = material_path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                blocks.append(self._image_block(material_path))
            elif suffix == ".txt":
                table_text = material_path.read_text(encoding="utf-8")
                blocks.append({"type": "text", "text": f"[Table: {filename}]\n{table_text}"})
            else:
                logger.warning(f"Unsupported material type, skipping: {filename}")
        return blocks

    def _build_sft_item(
        self,
        item: dict,
        answer: str,
        dataset_name: str,
        materials_dir: Path,
    ) -> dict:
        """Wrap a (question, GPT answer) pair into the Qwen-VL SFT schema."""
        question = item.get("question", "")
        material_filename = item.get("material")

        human_text = question
        image_paths: list[str] = []

        if material_filename:
            suffix = Path(material_filename).suffix.lower()
            material_path = materials_dir / material_filename
            if suffix in IMAGE_SUFFIXES:
                if material_path.exists():
                    image_paths.append(f"{dataset_name}/materials/{material_filename}")
                    human_text = f"{question}\n<image>"
                else:
                    logger.warning(f"Material image not found, omitting: {material_path}")
            elif suffix == ".txt":
                if material_path.exists():
                    table_text = material_path.read_text(encoding="utf-8")
                    human_text = f"{question}\n\n[Table: {material_filename}]\n{table_text}"
                else:
                    logger.warning(f"Material table not found, omitting: {material_path}")
            else:
                logger.warning(f"Unsupported material type, omitting: {material_filename}")

        sft_item: dict = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": answer},
            ],
            "source_chunk": item.get("source_chunk"),
            "type": item.get("type"),
        }
        if image_paths:
            sft_item["image"] = image_paths
        if material_filename:
            sft_item["material"] = material_filename
        return sft_item

    @staticmethod
    def _image_block(image_path: Path) -> dict:
        """Build an OpenAI image_url content block from a local image file."""
        mime = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
        b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        }

    @staticmethod
    def _human_text(sft_item: dict) -> str:
        """Extract the human turn text from an SFT item's conversations list."""
        for turn in sft_item.get("conversations", []):
            if turn.get("from") == "human":
                return turn.get("value", "")
        return ""

    @staticmethod
    def _load_json(path: Path) -> list | dict:
        """Load a JSON file and return its contents."""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(path: Path, data: list | dict) -> None:
        """Write data as JSON to path, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

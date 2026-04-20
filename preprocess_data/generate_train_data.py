"""Generate Qwen-VL SFT training data by producing answers for train QA pairs via GPT-5.1.

Recursively scans DEFAULT_EVAL_QA_DIR for `*_train.json` files, resolves each
question's source chunk from DEFAULT_PROCESSED_DIR, and asks GPT-5.1 to answer
it using the chunk text plus any <material:FILENAME> references embedded in
the chunk (images and tables) -- mirroring how `amba_eval_qa.py` passes
materials to the QA generator.

Each saved item follows the Qwen-VL finetune schema consumed by
`qwen-vl-finetune/qwenvl/data/data_processor.py::_build_messages`:

    {
      "conversations": [
        {"from": "human", "value": "<question> [<image>]"},
        {"from": "gpt",   "value": "<GPT answer>"}
      ],
      "image": ["<dataset>/materials/<filename>.jpg"],  # optional
      "source_chunk": "322.md",                          # metadata
      "type": "material" | "concept" | "reasoning",
      "material": "322_316.jpg"                          # metadata (optional)
    }

Image paths are relative to DEFAULT_PROCESSED_DIR, so the training config
should point `data_path` at that directory. Table (.txt) materials are
inlined into the human message since Qwen-VL only handles images/videos as
media.
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


DEFAULT_PROCESSED_DIR = Path(
    "data/datasets/amba_document/processed"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/datasets/amba_document/sft_data"
)
DEFAULT_MODEL = "gpt-5.1"

MATERIAL_TAG_PATTERN = re.compile(r"<material:([\w\-._]+)>")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

SYSTEM_PROMPT = """You are an expert on the AMBA (Advanced Microcontroller Bus Architecture) specifications.

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


logger = logging.getLogger(__name__)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def image_content_block(image_path: Path) -> dict:
    mime = MIME_MAP.get(image_path.suffix.lower(), "image/jpeg")
    b64 = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{b64}",
            "detail": "high",
        },
    }


def build_material_content(
    material_filenames: list[str],
    materials_dir: Path,
) -> list[dict]:
    """Build OpenAI content blocks for each material referenced in the chunk."""
    blocks: list[dict] = []
    for filename in material_filenames:
        material_path = materials_dir / filename
        if not material_path.exists():
            logger.warning(f"Material not found, skipping: {material_path}")
            continue
        suffix = material_path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            blocks.append(image_content_block(material_path))
        elif suffix == ".txt":
            table_text = material_path.read_text(encoding="utf-8")
            blocks.append(
                {
                    "type": "text",
                    "text": f"[Table: {filename}]\n{table_text}",
                }
            )
        else:
            logger.warning(f"Unsupported material type, skipping: {filename}")
    return blocks


def build_messages(
    question: str,
    chunk_text: str,
    materials_dir: Path,
) -> list[dict]:
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
    
    user_content.extend(build_material_content(material_filenames, materials_dir))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    chunk_text: str,
    materials_dir: Path,
) -> str:
    
    messages = build_messages(question, chunk_text, materials_dir)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def build_sft_item(
    item: dict,
    answer: str,
    dataset_name: str,
    materials_dir: Path,
) -> dict:
    """Wrap a (question, GPT answer) pair into the Qwen-VL SFT schema.

    Image-type materials become an <image> placeholder + entry in `image`
    (path relative to processed_dir). Table (.txt) materials are inlined
    as text at the end of the human message. Metadata fields (source_chunk,
    type, material) are preserved for traceability; the trainer ignores them.
    """
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


def process_train_file(
    client: OpenAI,
    model: str,
    train_file: Path,
    dataset_name: str,
    chunks_dir: Path,
    materials_dir: Path,
    output_file: Path,
) -> None:
    items = load_json(train_file)
    if not isinstance(items, list):
        logger.warning(f"Unexpected format (not a list): {train_file}")
        return

    # Resume: load any already-generated results and skip those questions.
    if output_file.exists():
        existing = load_json(output_file)
        results: list[dict] = existing if isinstance(existing, list) else []
    else:
        results = []
    done_keys: set[tuple[str, str]] = {
        (r.get("source_chunk", ""), _human_text_of(r)) for r in results
    }

    chunk_cache: dict[str, str] = {}

    desc = f"{train_file.parent.name}/{train_file.name}"
    for item in tqdm(items, desc=desc, leave=False):
        question = item.get("question", "")
        source_chunk = item.get("source_chunk")
        if not question or not source_chunk:
            logger.warning(f"Skipping item missing question/source_chunk: {item}")
            continue

        # Use (source_chunk, question) as the resume key even though we store
        # the transformed human_text; question uniquely identifies the item.
        resume_key_candidates = {
            (source_chunk, question),
            (source_chunk, f"{question}\n<image>"),
        }
        if done_keys & resume_key_candidates:
            continue

        if source_chunk not in chunk_cache:
            chunk_path = chunks_dir / source_chunk
            if not chunk_path.exists():
                logger.warning(f"Chunk not found, skipping item: {chunk_path}")
                continue
            chunk_cache[source_chunk] = chunk_path.read_text(encoding="utf-8")
        chunk_text = chunk_cache[source_chunk]

        try:
            answer = generate_answer(
                client=client,
                model=model,
                question=question,
                chunk_text=chunk_text,
                materials_dir=materials_dir,
            )
        except Exception as e:
            logger.error(f"GPT call failed for {source_chunk}: {e}")
            answer = ""

        sft_item = build_sft_item(item, answer, dataset_name, materials_dir)
        results.append(sft_item)
        done_keys.add((source_chunk, _human_text_of(sft_item)))

        # Save incrementally so progress is preserved on crash / Ctrl-C.
        save_json(output_file, results)
    logger.info(f"Wrote {len(results)} items -> {output_file}")


def _human_text_of(sft_item: dict) -> str:
    for turn in sft_item.get("conversations", []):
        if turn.get("from") == "human":
            return turn.get("value", "")
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SFT answers for AMBA train QA files using GPT-5.1."
    )
    parser.add_argument(
        "train_files",
        nargs="+",
        type=Path,
        help=(
            "One or more train JSON files, e.g. "
            "data/datasets/amba_document/eval_qa/<dataset>/<name>_train.json"
        ),
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even when the output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    load_dotenv()
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Pass --api-key or export the env var.")
        sys.exit(1)

    if not args.processed_dir.exists():
        raise FileNotFoundError(f"processed dir not found: {args.processed_dir}")

    client = OpenAI(api_key=api_key)

    for train_file in args.train_files:
        train_file = train_file.resolve()
        if not train_file.exists():
            logger.warning(f"Train file not found, skipping: {train_file}")
            continue

        # Expected layout: .../eval_qa/<dataset_name>/<filename>_train.json
        dataset_name = train_file.parent.name
        chunks_dir = args.processed_dir / dataset_name / "chunks"
        materials_dir = args.processed_dir / dataset_name / "materials"
        if not chunks_dir.exists():
            logger.warning(f"chunks dir missing, skipping: {chunks_dir}")
            continue
        if not materials_dir.exists():
            logger.warning(f"materials dir missing, skipping: {materials_dir}")
            continue

        output_file = args.output_dir / dataset_name / train_file.name

        # --overwrite wipes the existing output so we regenerate from scratch.
        # Without --overwrite we resume: process_train_file skips items whose
        # (source_chunk, question) pair is already present in the output.
        if args.overwrite and output_file.exists():
            output_file.unlink()

        logger.info(f"Processing {train_file}")
        process_train_file(
            client=client,
            model=args.model,
            train_file=train_file,
            dataset_name=dataset_name,
            chunks_dir=chunks_dir,
            materials_dir=materials_dir,
            output_file=output_file,
        )


if __name__ == "__main__":
    main()

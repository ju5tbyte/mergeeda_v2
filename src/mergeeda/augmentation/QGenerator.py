"""QA generation module using OpenAI GPT for text chunk question generation."""

import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You generate questions from a text chunk in three categories:
- concept: Factual question about a single concept or term.
- material: Questions that require interpreting a provided figure or table in context; avoid simple visual lookup (e.g., 'what is written here') and instead design questions that demand reasoning, such as inferring relationships, constraints, or behaviors implied by the figure/table with respect to the given text chunk. The referenced image will be provided at inference time.
- reasoning: Multi-step analytical question that synthesizes multiple ideas.

Rules:
1. All questions must be SELF-CONTAINED — fully understandable without the source text. Never reference "the passage", "section X", "the author", etc. The only exception: material questions may reference the figure/table with generic phrasing such as "According to the figure," or "Based on the table," — never use specific figure/table numbers or filenames in the question text.
2. Figures/tables appear as <material:FILENAME> tags in the text. For material questions, put the exact FILENAME in the "material" field. Each material question must reference exactly ONE figure/table; do not combine multiple figures/tables into a single question.
3. If no <material:...> tag exists, skip material questions entirely.
4. Dynamically decide how many questions to generate:
   - If the text is a table of contents, index, or similarly structure-only content with no substantive information, return an empty array [].
   - For typical text chunks, generate at least 1 question per applicable category.
   - For content-rich text (dense concepts, multiple figures/tables, or complex arguments), generate MULTIPLE questions per category as needed to adequately cover the material. For example, if the text contains 3 figures, generate up to 3 separate material questions, one per figure.

If materials are present, they are provided after the text chunk — tables as labeled text blocks ([Table: FILENAME]) and images as inline images — both in the order they appear in the text.

Output ONLY a JSON array, no other text:
[
  {"type":"concept","question":"..."},
  {"type":"material","question":"...","material":"FILENAME"},
  {"type":"reasoning","question":"..."}
]"""

SYSTEM_PROMPT_PREV = """You generate questions from a text chunk in three categories:
- concept: Factual question about a single concept or term.
- material: Question requiring interpretation of a specific figure/table. The referenced image will be provided at inference time.
- reasoning: Multi-step analytical question that synthesizes multiple ideas.

Rules:
1. All questions must be SELF-CONTAINED — fully understandable without the source text. Never reference "the passage", "section X", "the author", etc. The only exception: material questions may reference the figure/table with generic phrasing such as "According to the figure," or "Based on the table," — never use specific figure/table numbers or filenames in the question text.
2. Figures/tables appear as <material:FILENAME> tags in the text. For material questions, put the exact FILENAME in the "material" field. Each material question must reference exactly ONE figure/table; do not combine multiple figures/tables into a single question.
3. If no <material:...> tag exists, skip material questions entirely.
4. Dynamically decide how many questions to generate:
   - If the text is a table of contents, index, or similarly structure-only content with no substantive information, return an empty array [].
   - For typical text chunks, generate at least 1 question per applicable category.
   - For content-rich text (dense concepts, multiple figures/tables, or complex arguments), generate MULTIPLE questions per category as needed to adequately cover the material. For example, if the text contains 3 figures, generate up to 3 separate material questions, one per figure.
   - The total number of questions across all categories must not exceed 3.

If materials are present, they are provided after the text chunk — tables as labeled text blocks ([Table: FILENAME]) and images as inline images — both in the order they appear in the text.

Output ONLY a JSON array, no other text:
[
  {"type":"concept","question":"..."},
  {"type":"material","question":"...","material":"FILENAME"},
  {"type":"reasoning","question":"..."}
]"""

_USER_PROMPT_PREFIX = "<text_chunk>\n"
_USER_PROMPT_SUFFIX = (
    "\n</text_chunk>\n\nGenerate questions. Return ONLY the JSON array."
)

MATERIAL_TAG_PATTERN = re.compile(r"<material:([\w\-._]+)>")


class QGenerator:
    """Generate QA pairs from a single text chunk using GPT."""

    def __init__(
        self,
        model: str = "gpt-5.1",
        api_key: str | None = None,
    ) -> None:
        """Initialize QGenerator with OpenAI client."""
        self.model = model
        self._client = OpenAI(api_key=api_key)
        logger.info(f"QGenerator initialized with model={model}")

    def generate(
        self,
        chunk_path: str | Path,
        materials_dir: str | Path,
    ) -> list[dict]:
        """Generate questions for a single text chunk markdown file.

        Reads the chunk file, detects any <material:FILENAME> tags, loads the
        corresponding files from materials_dir, and calls the GPT API with the
        text (and images/tables as multimodal content where applicable).
        Returns a list of question dicts, each augmented with a source_chunk field.
        """
        chunk_path = Path(chunk_path)
        materials_dir = Path(materials_dir)

        text = chunk_path.read_text(encoding="utf-8")
        source_chunk = chunk_path.name
        logger.info(f"Generating questions for chunk: {source_chunk}")

        material_filenames = MATERIAL_TAG_PATTERN.findall(text)
        logger.debug(
            f"Found {len(material_filenames)} material references in {source_chunk}"
        )

        messages = self._build_messages(text, material_filenames, materials_dir)
        raw_questions = self._call_api(messages)

        questions = self._parse_response(raw_questions)
        for q in questions:
            q["source_chunk"] = source_chunk

        logger.info(f"Generated {len(questions)} questions for {source_chunk}")
        return questions

    def _build_messages(
        self,
        text: str,
        material_filenames: list[str],
        materials_dir: Path,
    ) -> list[dict]:
        """Build the OpenAI messages payload with text and optional material content."""
        user_content: list[dict] = [
            {
                "type": "text",
                "text": _USER_PROMPT_PREFIX + text + _USER_PROMPT_SUFFIX,
            }
        ]

        for filename in material_filenames:
            material_path = materials_dir / filename
            if not material_path.exists():
                logger.warning(
                    f"Material file not found, skipping: {material_path}"
                )
                continue

            suffix = material_path.suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                user_content.append(
                    self._image_content_block(material_path, suffix)
                )
            elif suffix == ".txt":
                table_text = material_path.read_text(encoding="utf-8")
                user_content.append(
                    {
                        "type": "text",
                        "text": f"[Table: {filename}]\n{table_text}",
                    }
                )
            else:
                logger.warning(
                    f"Unsupported material type, skipping: {filename}"
                )
                

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _image_content_block(self, image_path: Path, suffix: str) -> dict:
        """Encode an image file to base64 and return an OpenAI image content block."""
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode(
            "utf-8"
        )
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}",
                "detail": "high",
            },
        }

    def _call_api(self, messages: list[dict]) -> str:
        """Call the OpenAI chat completion API and return the response text."""
        logger.debug(f"Calling OpenAI API with model={self.model}")
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def _parse_response(self, response_text: str) -> list[dict]:
        """Parse the JSON array from GPT response."""
        try:
            questions = json.loads(response_text.strip())
            if not isinstance(questions, list):
                logger.warning(
                    "GPT response is not a JSON array, returning empty list"
                )
                return []
            return questions
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse GPT response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return []

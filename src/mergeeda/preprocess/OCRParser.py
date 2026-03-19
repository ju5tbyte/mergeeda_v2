"""PDF OCR parsing module using DeepSeek-OCR with vLLM."""

import logging
import re
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

logger = logging.getLogger(__name__)


class OCRParser:
    """Parse PDF documents to chunked markdown using DeepSeek-OCR."""

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        dpi: int = 300,
    ) -> None:
        """Initialize OCR parser with vLLM model."""
        self.model_name = model_name
        self.dpi = dpi
        self.llm: LLM | None = None
        logger.info(f"OCRParser initialized with model={model_name}, dpi={dpi}")

    def _init_model(self) -> None:
        """Initialize vLLM model lazily."""
        if self.llm is None:
            logger.info("Loading DeepSeek-OCR model with vLLM...")
            self.llm = LLM(
                model=self.model_name,
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor],
            )
            logger.info("Model loaded successfully")

    def parse_pdf(
        self,
        input_pdf: str | Path,
        output_dir: str | Path,
        chunk_level: int = 3,
        level_patterns: list[str] | None = None,
    ) -> None:
        """Parse PDF document and save chunked markdown with extracted images."""
        input_path = Path(input_pdf)
        output_path = Path(output_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_path}")

        logger.info(f"Starting PDF parsing: {input_path}")
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Chunk level: {chunk_level}")

        # Create output directories
        chunks_dir = output_path / "chunks"
        materials_dir = output_path / "materials"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        materials_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert PDF to images
        logger.info("Converting PDF to images...")
        page_images = self._convert_pdf_to_images(input_path)
        logger.info(f"Converted {len(page_images)} pages")

        # Step 2: Initialize model
        self._init_model()

        # Step 3: OCR all pages
        logger.info("Running OCR on all pages...")
        ocr_results = self._ocr_pages(page_images)

        # Step 4: Extract materials (images and tables) and replace tags
        logger.info("Extracting materials from OCR results...")
        markdown_text, material_metadata = self._extract_and_replace_materials(
            ocr_results, page_images, materials_dir
        )

        # Step 5: Chunk markdown by heading level
        logger.info(f"Chunking markdown by level {chunk_level}...")
        chunks = self._chunk_markdown(
            markdown_text, chunk_level, level_patterns
        )
        logger.info(f"Created {len(chunks)} chunks")

        # Step 6: Save chunks with proper material references
        logger.info("Saving chunks...")
        self._save_chunks(chunks, chunks_dir, material_metadata)

        logger.info("PDF parsing completed successfully")

    def _convert_pdf_to_images(self, pdf_path: Path) -> list[Image.Image]:
        """Convert PDF pages to PIL images."""
        return convert_from_path(str(pdf_path), dpi=self.dpi)

    def _ocr_pages(self, images: list[Image.Image]) -> list[str]:
        """Run OCR on all pages using vLLM batch inference."""
        if self.llm is None:
            raise RuntimeError("Model not initialized")

        prompt = "<image>\n<|grounding|>Convert the document to markdown."

        model_inputs = []
        for img in images:
            model_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": img.convert("RGB")},
                }
            )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
        )

        outputs = self.llm.generate(model_inputs, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def _extract_and_replace_materials(
        self,
        ocr_results: list[str],
        page_images: list[Image.Image],
        materials_dir: Path,
    ) -> tuple[str, dict[str, tuple[int, int, str]]]:
        """Extract materials (images, tables) from OCR results and replace with tags.

        DeepSeek-OCR returns structured output with ref-det tags:
        <|ref|>TYPE<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

        Strategy:
        1. Parse all ref-det tags from original text
        2. Process in REVERSE order to avoid index shifting
        3. For images: crop and save from page image, replace tag with <material:...>
        4. For tables: find <table>...</table> block, save as text, replace with <material:...>
        5. For others (title, text, etc.): just remove the tag, keep content

        Returns:
            Tuple of (processed markdown text, material metadata mapping)
            material_metadata maps material filename to (chunk_id, material_id, material_type)
        """
        # Pattern to match: <|ref|>TYPE<|/ref|><|det|>[[...]]<|/det|>
        tag_pattern = re.compile(
            r"<\|ref\|>(\w+)<\|/ref\|><\|det\|>\[\[([^\]]+(?:\],\s*\[[^\]]+)*)\]\]<\|/det\|>",
            re.IGNORECASE,
        )

        combined_markdown = ""
        material_counter = 1
        material_metadata: dict[str, tuple[int, int, str]] = {}

        for page_idx, ocr_text in enumerate(ocr_results):
            page_num = page_idx + 1
            logger.info(f"Processing page {page_num}/{len(ocr_results)}")

            # Parse all tags with their positions
            tags = []
            for match in tag_pattern.finditer(ocr_text):
                ref_type = match.group(1).lower()
                coords_str = match.group(2)

                # Parse first bbox (handle both single and multiple bboxes)
                bbox_groups = coords_str.split("], [")
                first_bbox_str = bbox_groups[0].replace("[", "").replace("]", "")

                try:
                    coords = [int(x.strip()) for x in first_bbox_str.split(",")]
                    if len(coords) != 4:
                        logger.warning(f"Invalid bbox format: {first_bbox_str}, skipping")
                        continue
                    x1, y1, x2, y2 = coords
                except ValueError as e:
                    logger.warning(f"Failed to parse bbox: {first_bbox_str}, error: {e}")
                    continue

                tags.append({
                    "type": ref_type,
                    "bbox": (x1, y1, x2, y2),
                    "start": match.start(),
                    "end": match.end(),
                })

            logger.debug(f"Found {len(tags)} ref-det tags on page {page_num}")

            # Process tags in REVERSE order
            processed_text = ocr_text

            for tag_info in reversed(tags):
                ref_type = tag_info["type"]
                x1, y1, x2, y2 = tag_info["bbox"]
                tag_start = tag_info["start"]
                tag_end = tag_info["end"]

                # Validate bounding box
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Invalid bbox for {ref_type}: [[{x1}, {y1}, {x2}, {y2}]]")
                    processed_text = processed_text[:tag_start] + processed_text[tag_end:]
                    continue

                # Handle material types
                if ref_type == "image":
                    material_filename = f"{material_counter}.jpg"
                    self._crop_and_save_image(
                        page_images[page_idx],
                        x1, y1, x2, y2,
                        materials_dir / material_filename,
                    )

                    # Replace tag with material reference
                    replacement = f"<material:{material_filename}>"
                    processed_text = (
                        processed_text[:tag_start]
                        + replacement
                        + processed_text[tag_end:]
                    )

                    material_metadata[material_filename] = (-1, material_counter, "image")
                    material_counter += 1

                elif ref_type == "table":
                    # Find <table>...</table> block near this tag
                    table_text, table_start, table_end = self._find_table_block(
                        processed_text, tag_start, tag_end
                    )

                    if table_text:
                        material_filename = f"{material_counter}.txt"
                        self._save_table_as_text(
                            table_text,
                            materials_dir / material_filename,
                        )

                        # Remove entire region (tag + table block)
                        remove_start = min(tag_start, table_start)
                        remove_end = max(tag_end, table_end)

                        replacement = f"<material:{material_filename}>"
                        processed_text = (
                            processed_text[:remove_start]
                            + replacement
                            + processed_text[remove_end:]
                        )

                        material_metadata[material_filename] = (-1, material_counter, "table")
                        material_counter += 1
                    else:
                        # No table block found, just remove tag
                        logger.warning(f"No <table> block found for table tag at position {tag_start}")
                        processed_text = processed_text[:tag_start] + processed_text[tag_end:]

                else:
                    # For title, sub_title, figure_title, text, etc.
                    # Just remove the tag, keep the content
                    processed_text = processed_text[:tag_start] + processed_text[tag_end:]

            combined_markdown += processed_text + "\n\n"

        return combined_markdown, material_metadata

    def _crop_and_save_image(
        self,
        page_image: Image.Image,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        output_path: Path,
    ) -> None:
        """Crop image using normalized coordinates and save as JPG.

        Coordinates are in 0-999 range and need to be denormalized.
        """
        width, height = page_image.size

        # Denormalize coordinates (0-999 -> pixel coordinates)
        pixel_x1 = int(x1 * width / 1000)
        pixel_y1 = int(y1 * height / 1000)
        pixel_x2 = int(x2 * width / 1000)
        pixel_y2 = int(y2 * height / 1000)

        # Crop and save
        cropped = page_image.crop((pixel_x1, pixel_y1, pixel_x2, pixel_y2))
        cropped.save(output_path, "JPEG", quality=95)
        logger.debug(f"Saved image: {output_path}")

    def _find_table_block(
        self, text: str, tag_start: int, tag_end: int
    ) -> tuple[str, int, int]:
        """Find <table>...</table> block immediately after the ref-det tag.

        Returns:
            Tuple of (table_content, block_start, block_end)
            If not found, returns ("", -1, -1)
        """
        # Get text after the tag
        remaining_text = text[tag_end:]

        # Skip whitespace/newlines to find <table> start
        stripped_start = len(remaining_text) - len(remaining_text.lstrip())
        stripped_text = remaining_text.lstrip()

        # Check if next non-whitespace content is <table>
        if not stripped_text.lower().startswith("<table>"):
            return "", -1, -1

        # Find matching </table>
        table_pattern = re.compile(r"^<table>(.*?)</table>", re.DOTALL | re.IGNORECASE)
        table_match = table_pattern.match(stripped_text)

        if table_match:
            # Calculate absolute positions
            block_start = tag_end + stripped_start
            block_end = block_start + table_match.end()
            table_content = table_match.group(1).strip()
            return table_content, block_start, block_end

        return "", -1, -1

    def _save_table_as_text(self, table_text: str, output_path: Path) -> None:
        """Save table content as text file."""
        output_path.write_text(table_text, encoding="utf-8")
        logger.debug(f"Saved table: {output_path}")

    def _chunk_markdown(
        self,
        markdown_text: str,
        chunk_level: int,
        level_patterns: list[str] | None = None,
    ) -> list[str]:
        """Chunk markdown by heading level based on section numbering.

        Args:
            markdown_text: Full markdown text
            chunk_level: Number of dots in section numbers to split on (e.g., 3 for "1.2.3")
            level_patterns: Custom regex patterns for each level (default: dot-separated numbers)

        Returns:
            List of markdown chunks
        """
        lines = markdown_text.split("\n")

        # Build default pattern if not provided
        if level_patterns is None:
            # Default: match patterns like "1.2.3" (level 3), "B3.1" (level 2), etc.
            # Count dots to determine level
            default_pattern = r"^#{1,6}\s+(?:[A-Z])?(\d+(?:\.\d+)*)"
            level_patterns = [default_pattern]

        # Pattern to extract section numbers
        heading_pattern = re.compile(level_patterns[0])

        chunks: list[list[str]] = []
        current_chunk: list[str] = []

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                # Extract section number (e.g., "1.2.3" or "3.1")
                section_num = match.group(1)
                # Count dots to determine level
                dots_count = section_num.count(".")
                current_level = dots_count + 1

                # If this is a heading at or above chunk_level, start new chunk
                if current_level <= chunk_level:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return ["\n".join(chunk) for chunk in chunks]

    def _save_chunks(
        self,
        chunks: list[str],
        chunks_dir: Path,
        material_metadata: dict[str, tuple[int, int, str]],
    ) -> None:
        """Save chunks to individual markdown files and update material references.

        Renames materials to follow chunk_id_material_id[_caption].ext pattern.
        """
        materials_dir = chunks_dir.parent / "materials"

        # First pass: assign chunk IDs to materials
        material_to_chunk: dict[str, int] = {}
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            # Find all material references in this chunk
            material_pattern = re.compile(r"<material:([\w\-._]+)>")
            for match in material_pattern.finditer(chunk_text):
                material_filename = match.group(1)
                if material_filename in material_metadata:
                    material_to_chunk[material_filename] = chunk_idx

        # Second pass: rename materials and update chunk text
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            # Find and replace material references
            material_pattern = re.compile(r"<material:([\w\-._]+)>")

            def replace_material_ref(match: re.Match) -> str:
                old_filename = match.group(1)
                if old_filename in material_metadata:
                    _, mat_id, _ = material_metadata[old_filename]

                    # Extract extension and caption suffix from old filename
                    old_path = Path(old_filename)
                    extension = old_path.suffix
                    stem = old_path.stem

                    # Check if old filename has caption suffix
                    # Pattern: {number}_{caption} or just {number}
                    if "_" in stem:
                        # Has caption: preserve it
                        parts = stem.split("_", 1)
                        caption_suffix = f"_{parts[1]}"
                    else:
                        caption_suffix = ""

                    # Build new filename: chunk_id_mat_id[_caption].ext
                    new_filename = (
                        f"{chunk_idx}_{mat_id}{caption_suffix}{extension}"
                    )

                    # Rename physical file
                    old_path_full = materials_dir / old_filename
                    new_path = materials_dir / new_filename
                    if old_path_full.exists() and not new_path.exists():
                        old_path_full.rename(new_path)
                        logger.debug(
                            f"Renamed {old_filename} -> {new_filename}"
                        )

                    return f"<material:{new_filename}>"
                return match.group(0)

            updated_chunk = material_pattern.sub(
                replace_material_ref, chunk_text
            )

            # Save chunk
            chunk_file = chunks_dir / f"{chunk_idx}.md"
            chunk_file.write_text(updated_chunk, encoding="utf-8")
            logger.debug(f"Saved chunk {chunk_idx}: {chunk_file}")

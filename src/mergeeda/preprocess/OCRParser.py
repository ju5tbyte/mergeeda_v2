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
        images_dir = output_path / "images"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Convert PDF to images
        logger.info("Converting PDF to images...")
        page_images = self._convert_pdf_to_images(input_path)
        logger.info(f"Converted {len(page_images)} pages")

        # Step 2: Initialize model
        self._init_model()

        # Step 3: OCR all pages
        logger.info("Running OCR on all pages...")
        ocr_results = self._ocr_pages(page_images)

        # Step 4: Extract images and replace tags
        logger.info("Extracting images from OCR results...")
        markdown_text, image_metadata = self._extract_and_replace_images(
            ocr_results, page_images, images_dir
        )

        # Step 5: Chunk markdown by heading level
        logger.info(f"Chunking markdown by level {chunk_level}...")
        chunks = self._chunk_markdown(
            markdown_text, chunk_level, level_patterns
        )
        logger.info(f"Created {len(chunks)} chunks")

        # Step 6: Save chunks with proper image references
        logger.info("Saving chunks...")
        self._save_chunks(chunks, chunks_dir, image_metadata)

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

    def _extract_and_replace_images(
        self,
        ocr_results: list[str],
        page_images: list[Image.Image],
        images_dir: Path,
    ) -> tuple[str, dict[str, tuple[int, int]]]:
        """Extract images from bounding boxes and replace tags with image references.

        Returns:
            Tuple of (processed markdown text, image metadata mapping)
            image_metadata maps image filename to (chunk_id, image_id)
        """
        # Pattern to match: <|ref|>image<|/ref|><|det|>[x1, y1, x2, y2]<|/det|>
        # Only match when ref tag contains "image" label
        pattern = re.compile(
            r"<\|ref\|>(image)<\|/ref\|><\|det\|>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]<\|/det\|>",
            re.IGNORECASE,
        )

        combined_markdown = ""
        image_counter = 1
        image_metadata: dict[str, tuple[int, int]] = {}

        for page_idx, ocr_text in enumerate(ocr_results):
            page_num = page_idx + 1
            logger.info(f"Processing page {page_num}/{len(ocr_results)}")

            # Find all matches
            matches = list(pattern.finditer(ocr_text))
            logger.debug(
                f"Found {len(matches)} bounding boxes on page {page_num}"
            )

            # Process matches in reverse to maintain string indices
            processed_text = ocr_text
            for match in reversed(matches):
                # ref_text is "image" (from the pattern)
                x1, y1, x2, y2 = map(int, match.groups()[1:])

                # Skip if bounding box looks invalid
                if x1 >= x2 or y1 >= y2:
                    logger.warning(
                        f"Invalid bounding box: [{x1}, {y1}, {x2}, {y2}]"
                    )
                    continue

                # Crop image from page
                image_filename = f"{image_counter}.jpg"
                self._crop_and_save_image(
                    page_images[page_idx],
                    x1,
                    y1,
                    x2,
                    y2,
                    images_dir / image_filename,
                )

                # Replace the entire match with image tag
                image_tag = f"<image:{image_filename}>"
                processed_text = (
                    processed_text[: match.start()]
                    + image_tag
                    + processed_text[match.end() :]
                )

                # Store metadata (will be updated with chunk_id later)
                image_metadata[image_filename] = (-1, image_counter)
                image_counter += 1

            combined_markdown += processed_text + "\n\n"

        return combined_markdown, image_metadata

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

        Coordinates are in 0-999 range and need to be denormalized to image dimensions.
        """
        width, height = page_image.size

        # Denormalize coordinates
        pixel_x1 = int(x1 * width / 1000)
        pixel_y1 = int(y1 * height / 1000)
        pixel_x2 = int(x2 * width / 1000)
        pixel_y2 = int(y2 * height / 1000)

        # Crop and save
        cropped = page_image.crop((pixel_x1, pixel_y1, pixel_x2, pixel_y2))
        cropped.save(output_path, "JPEG", quality=95)
        logger.debug(f"Saved cropped image: {output_path}")

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
        image_metadata: dict[str, tuple[int, int]],
    ) -> None:
        """Save chunks to individual markdown files and update image references.

        Also renames images to follow chunk_id_image_id.jpg pattern.
        """
        images_dir = chunks_dir.parent / "images"

        # First pass: assign chunk IDs to images
        image_to_chunk: dict[str, int] = {}
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            # Find all image references in this chunk
            image_pattern = re.compile(r"<image:([\w\-.]+)>")
            for match in image_pattern.finditer(chunk_text):
                image_filename = match.group(1)
                if image_filename in image_metadata:
                    image_to_chunk[image_filename] = chunk_idx

        # Second pass: rename images and update chunk text
        for chunk_idx, chunk_text in enumerate(chunks, start=1):
            # Find and replace image references
            image_pattern = re.compile(r"<image:([\w\-.]+)>")

            def replace_image_ref(match: re.Match) -> str:
                old_filename = match.group(1)
                if old_filename in image_metadata:
                    _, img_id = image_metadata[old_filename]
                    new_filename = f"{chunk_idx}_{img_id}.jpg"

                    # Rename physical file
                    old_path = images_dir / old_filename
                    new_path = images_dir / new_filename
                    if old_path.exists() and not new_path.exists():
                        old_path.rename(new_path)
                        logger.debug(
                            f"Renamed {old_filename} -> {new_filename}"
                        )

                    return f"<image:{new_filename}>"
                return match.group(0)

            updated_chunk = image_pattern.sub(replace_image_ref, chunk_text)

            # Save chunk
            chunk_file = chunks_dir / f"{chunk_idx}.md"
            chunk_file.write_text(updated_chunk, encoding="utf-8")
            logger.debug(f"Saved chunk {chunk_idx}: {chunk_file}")

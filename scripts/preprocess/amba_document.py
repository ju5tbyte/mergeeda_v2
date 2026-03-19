"""Script to parse AMBA document PDF using OCRParser."""

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from mergeeda.preprocess import OCRParser

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="amba_document",
)
def main(cfg: DictConfig) -> None:
    """Parse AMBA document PDF to chunked markdown."""
    logger.info("Starting AMBA document parsing")
    logger.info(f"Input PDF: {cfg.input_pdf}")
    logger.info(f"Output directory: {cfg.output_dir}")

    # Resolve paths relative to project root (original working directory)
    original_cwd = Path(get_original_cwd())
    input_path = original_cwd / cfg.input_pdf
    output_path = original_cwd / cfg.output_dir

    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    # Initialize OCR parser
    parser = OCRParser(
        model_name=cfg.model.name,
        dpi=cfg.model.dpi,
    )

    # Parse PDF with chunking configuration
    parser.parse_pdf(
        input_pdf=input_path,
        output_dir=output_path,
        chunk_level=cfg.chunking.level,
        level_patterns=cfg.chunking.level_patterns,
    )

    logger.info("AMBA document parsing completed successfully")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"  - Chunks: {output_path / 'chunks'}")
    logger.info(f"  - Materials: {output_path / 'materials'}")


if __name__ == "__main__":
    main()

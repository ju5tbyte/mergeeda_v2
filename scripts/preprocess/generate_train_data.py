"""Script to generate SFT training data for AMBA train QA files using GPT."""

import logging
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from mergeeda.preprocess import TrainDataGenerator

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="generate_train_data",
)
def main(cfg: DictConfig) -> None:
    """Generate GPT answers for each train QA file and save as SFT JSON."""
    logger.info("Starting SFT training data generation")
    logger.info(f"Processed dir: {cfg.processed_dir}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Train files: {list(cfg.train_files)}")

    original_cwd = Path(get_original_cwd())
    processed_dir = original_cwd / cfg.processed_dir
    output_dir = original_cwd / cfg.output_dir

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {processed_dir}")

    api_key: str | None = cfg.model.api_key or os.environ.get("OPENAI_API_KEY")

    generator = TrainDataGenerator(model=cfg.model.name, api_key=api_key)

    for train_file_rel in cfg.train_files:
        train_file = (original_cwd / train_file_rel).resolve()
        if not train_file.exists():
            logger.warning(f"Train file not found, skipping: {train_file}")
            continue

        # Expected layout: .../eval_qa/<dataset_name>/<filename>_train.json
        dataset_name = train_file.parent.name
        chunks_dir = processed_dir / dataset_name / "chunks"
        materials_dir = processed_dir / dataset_name / "materials"

        if not chunks_dir.exists():
            logger.warning(f"Chunks dir missing, skipping: {chunks_dir}")
            continue
        if not materials_dir.exists():
            logger.warning(f"Materials dir missing, skipping: {materials_dir}")
            continue

        output_file = output_dir / dataset_name / train_file.name
        logger.info(f"Processing {train_file}")

        generator.generate(
            train_file=train_file,
            dataset_name=dataset_name,
            chunks_dir=chunks_dir,
            materials_dir=materials_dir,
            output_file=output_file,
            overwrite=cfg.overwrite,
        )

    logger.info("SFT training data generation completed")


if __name__ == "__main__":
    main()

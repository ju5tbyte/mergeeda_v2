"""Script to merge per-dataset SFT JSON files into a single annotation file."""

import logging
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from mergeeda.preprocess import TrainDataMerger

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/preprocess",
    config_name="merge_train_data",
)
def main(cfg: DictConfig) -> None:
    """Merge all SFT JSON files under sft_data_dir into a single file."""
    logger.info("Starting SFT training data merge")
    logger.info(f"SFT data dir: {cfg.sft_data_dir}")
    logger.info(f"Output path: {cfg.output_path}")

    original_cwd = Path(get_original_cwd())
    sft_data_dir = original_cwd / cfg.sft_data_dir
    output_path = original_cwd / cfg.output_path

    merger = TrainDataMerger()
    merger.merge(sft_data_dir=sft_data_dir, output_path=output_path)

    logger.info("SFT training data merge completed")
    logger.info(f"Merged file saved to: {output_path}")


if __name__ == "__main__":
    main()

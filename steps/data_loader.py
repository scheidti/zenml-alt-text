import logging

from zenml import step
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


@step
def load_data(
    dataset: str, image_row: str = "image", split: str = "validation"
) -> Dataset:
    data = load_dataset(dataset, split=split)

    cols_to_drop = [col for col in data.column_names if col != image_row]
    data = data.remove_columns(cols_to_drop)

    logger.info(f"Loaded dataset {dataset} with split {split}.")
    return data

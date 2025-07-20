from zenml import step
from zenml.logger import get_logger
from datasets import load_dataset, Dataset

logger = get_logger(__name__)


@step
def load_data(
    dataset: str, image_row: str = "image", split: str = "validation"
) -> Dataset:
    """Load a dataset and filter to keep only the specified image column.

    This ZenML step loads a dataset from Hugging Face datasets and removes all
    columns except for the specified image column, preparing the data for
    further processing in the pipeline.

    Args:
        dataset: Name of the dataset to load from Hugging Face datasets.
        image_row: Name of the column containing image data. Defaults to "image".
        split: Dataset split to load (e.g., "train", "validation", "test").
               Defaults to "validation".

    Returns:
        Dataset: A Hugging Face Dataset containing only the specified image column.
    """
    data = load_dataset(dataset, split=split)

    cols_to_drop = [col for col in data.column_names if col != image_row]
    data = data.remove_columns(cols_to_drop)

    logger.info(f"Loaded dataset {dataset} with split {split}.")
    return data

from zenml import step
from zenml.logger import get_logger
from zenml.client import Client
from datasets import load_dataset, Dataset, DatasetDict

logger = get_logger(__name__)

client = Client()


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


@step
def load_training_data(
    pipeline_name: str = "alt_text_batch_processing_pipeline",
    step_name: str = "add_batch_results_to_dataset",
    test_size: int = 1000,
    validation_size: int = 1000,
    seed: int = 42,
) -> DatasetDict:
    run = client.get_pipeline(pipeline_name).last_successful_run
    hf_dataset_id: str = run.steps[step_name].output.load()
    data: DatasetDict = load_dataset(hf_dataset_id)

    combined_dataset = None
    for _, split_data in data.items():
        if combined_dataset is None:
            combined_dataset = split_data
        else:
            combined_dataset = combined_dataset.concatenate(split_data)

    train_val_split = combined_dataset.train_test_split(
        test_size=test_size, shuffle=True, seed=seed
    )
    test_dataset = train_val_split["test"]
    train_val_dataset = train_val_split["train"]

    train_val_split = train_val_dataset.train_test_split(
        test_size=validation_size, shuffle=True, seed=seed
    )
    train_dataset = train_val_split["train"]
    validation_dataset = train_val_split["test"]

    result = DatasetDict(
        {"train": train_dataset, "test": test_dataset, "validation": validation_dataset}
    )

    logger.info(
        f"Created splits - Train: {len(train_dataset)}, Test: {len(test_dataset)}, Validation: {len(validation_dataset)}"
    )
    return result

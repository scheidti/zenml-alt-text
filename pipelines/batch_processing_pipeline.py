from pathlib import Path
from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from steps.batch_tasks import (
    load_batch_task_list,
    wait_and_update_batch,
    download_batch_results,
)

client = Client()
logger = get_logger(__name__)


@pipeline(name="alt_text_data_preparation_pipeline")
def batch_processing_pipeline() -> list[Path]:
    tasks = load_batch_task_list()
    worked_tasks = wait_and_update_batch(tasks)
    result_files = download_batch_results(worked_tasks)
    return result_files

from zenml import pipeline
from zenml.client import Client
from zenml.logger import get_logger

from steps.batch_tasks import (
    load_batch_task_list,
    wait_and_update_batch,
    download_batch_results,
    add_batch_results_to_dataset,
)

client = Client()
logger = get_logger(__name__)


@pipeline(name="alt_text_batch_processing_pipeline")
def batch_processing_pipeline() -> str:
    tasks = load_batch_task_list()
    worked_tasks = wait_and_update_batch(tasks)
    result_files = download_batch_results(worked_tasks)
    hf_repo_id = add_batch_results_to_dataset(result_files=result_files)
    return hf_repo_id

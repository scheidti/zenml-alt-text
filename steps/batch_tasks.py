from pathlib import Path
from openai import OpenAI
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from time import sleep
from datasets import Dataset
import pandas as pd

from utils.pydantic_models import BatchFileTaskList

client = Client()
openai = OpenAI()
logger = get_logger(__name__)


@step
def load_batch_task_list(
    pipeline_name: str = "alt_text_data_preparation_pipeline",
    step_name: str = "upload_files_to_openai",
) -> BatchFileTaskList:
    logger.info(
        f"Loading batch task list from pipeline: {pipeline_name}, step: {step_name}"
    )
    run = client.get_pipeline(pipeline_name).last_successful_run
    task_list = run.steps[step_name].output.load()
    logger.info(f"Loaded task list with {len(task_list.tasks)} tasks.")
    return task_list


@step()
def wait_and_update_batch(
    task_list: BatchFileTaskList,
    wait_seconds: int = 60,
) -> BatchFileTaskList:
    for task in task_list.tasks:
        if task.batch_id is not None:
            logger.info(f"Task {task.file_id} already has a batch ID {task.batch_id}.")
            batch = openai.batches.retrieve(task.batch_id)
            task.status = batch.status
            task.result_file_id = getattr(batch, "output_file_id", None)
            logger.info(
                f"Updated task {task.file_id} with existing batch ID {task.batch_id}."
            )
            continue

        for batch in openai.batches.list():
            if batch.input_file_id == task.file_id:
                task.status = batch.status
                task.result_file_id = getattr(batch, "output_file_id", None)
                task.batch_id = batch.id
                logger.info(
                    f"Found existing OpenAI Batch {batch.id} for task {task.file_id}."
                )
                break

    do_not_work_status = [
        "validating",
        "finalizing",
        "completed",
        "cancelling",
        "cancelled",
    ]
    stop_working_status = ["failed", "expired", "cancelled", "completed"]

    for task in task_list.tasks:
        if task.status in do_not_work_status:
            logger.info(f"Task {task.file_id} is in status {task.status}, ignoring.")
            continue

        logger.info(f"Starting OpenAI Batch for task {task.file_id}.")
        if task.status != "in_progress":
            # Only create a new batch if the task is not already in progress
            job = openai.batches.create(
                input_file_id=task.file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            task.batch_id = job.id
            task.status = job.status
            logger.info(
                f"Created OpenAI Batch with ID {task.batch_id} for task {task.file_id}."
            )

        # Wait for the task to complete or reach a terminal status
        while task.status not in stop_working_status:
            sleep(wait_seconds)
            job = openai.batches.retrieve(task.batch_id)
            task.status = job.status
            task.result_file_id = getattr(job, "output_file_id", None)
            logger.info(f"Updated task {task.file_id} status to {task.status}.")

    for task in task_list.tasks:
        print(f"{task}")

    return task_list


@step()
def download_batch_results(
    task_list: BatchFileTaskList,
) -> list[Path]:
    logger.info("Downloading batch results for all tasks.")
    downloaded_files = []

    for task in task_list.tasks:
        if task.status == "completed" and task.result_file_id:
            logger.info(f"Downloading result file for task {task.file_id}.")
            file_content = openai.files.content(task.result_file_id)
            file_path = Path(f"batches/{task.file_id}_result.jsonl")

            with open(file_path, "wb") as f:
                f.write(file_content.read())

            downloaded_files.append(file_path)
            logger.info(f"Downloaded result file to {file_path}.")
        else:
            logger.warning(f"No result file for task {task.file_id}.")

    logger.info(f"Downloaded {len(downloaded_files)} result files.")
    return downloaded_files


@step()
def add_batch_results_to_dataset(
    result_files: list[Path],
    pipeline_name: str = "alt_text_data_preparation_pipeline",
    step_name: str = "load_data",
    hf_repo_id: str = "scheidti/vqav2-small-alt-text",
) -> str:
    logger.info("Adding batch results to dataset.")
    run = client.get_pipeline(pipeline_name).last_successful_run
    dataset: Dataset = run.steps[step_name].output.load()
    alt_text_mapping = {}

    for file_path in result_files:
        logger.info(f"Adding results from {file_path} to dataset.")
        data = pd.read_json(file_path, lines=True)
        for _, row in data.iterrows():
            status_code = row.get("response").get("status_code")
            row_id = int(row.get("custom_id").replace("row_", ""))

            if status_code != 200:
                logger.warning(
                    f"Skipping row {row_id} due to status code {status_code}."
                )
                continue

            alt_text = (
                row.get("response")
                .get("body")
                .get("choices")[0]
                .get("message")
                .get("content")
            )

            if not alt_text:
                logger.warning(f"No alt text generated for row {row_id}.")
                continue

            alt_text_mapping[row_id] = alt_text

    def add_alt_text(example, idx):
        example["alt_text"] = alt_text_mapping.get(idx, "")
        return example

    dataset = dataset.map(add_alt_text, with_indices=True)
    logger.info(f"Attempting to push dataset to Hugging Face Hub: {hf_repo_id}")
    commit_info = dataset.push_to_hub(hf_repo_id)
    logger.info(f"Dataset pushed to Hugging Face Hub at {commit_info.repo_url}.")

    return commit_info.repo_url.repo_id

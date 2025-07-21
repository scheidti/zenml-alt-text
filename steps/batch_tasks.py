from openai import OpenAI
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from time import sleep

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


@step(enable_cache=False)
def wait_and_update_batch(
    task_list: BatchFileTaskList,
    wait_seconds: int = 60,
) -> BatchFileTaskList:
    for task in task_list.tasks:
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

import time
from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

from services.openai_batch_service import start_openai_batch_service
from utils.pydantic_models import BatchFileTaskList, BatchFileTask

client = Client()
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
    # task: BatchFileTask | None,
    wait_time: int = 300,
) -> BatchFileTaskList:
    for task in task_list.tasks:
        if task is None:
            logger.info("No task or service to wait for.")
            return task_list

        logger.info(f"Waiting for task {task.file_id} to complete.")
        service = start_openai_batch_service(task.file_id, name=f"batch_{task.file_id}")

        while service.is_running:
            time.sleep(wait_time)
            service.update_status()

        logger.info(
            f"Task {task.file_id} completed with status: {service.status.openai_state}"
        )

        if service.status.openai_state != "completed":
            task.status = "failed"
        else:
            task.status = "done"
            task.result_file_id = service.status.result_file_id

    return task_list    

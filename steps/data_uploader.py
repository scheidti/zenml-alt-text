from pathlib import Path
from zenml import step
from zenml.logger import get_logger
from openai import OpenAI

from utils.pydantic_models import BatchFileTaskList, BatchFileTask

logger = get_logger(__name__)
client = OpenAI()


@step
def upload_files_to_openai(files: list[Path]) -> BatchFileTaskList:
    tasks = []
    logger.info(f"Uploading {len(files)} files to OpenAI for batch processing")

    for path in files:
        logger.info(f"Uploading file: {path}")
        batch_input_file = client.files.create(file=open(path, "rb"), purpose="batch")
        task = BatchFileTask(
            file_id=batch_input_file.id,
            path=str(path),
            status="pending",
            result_file_id=None,
            batch_id=None,
        )
        logger.info(f"File uploaded with ID: {task.file_id}")
        tasks.append(task)

    return BatchFileTaskList(tasks=tasks)

from typing import List, Literal
from pydantic import BaseModel


class BatchFileTask(BaseModel):
    file_id: str
    path: str
    status: Literal[
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    ] = "validating"
    result_file_id: str | None = None
    batch_id: str | None = None


class BatchFileTaskList(BaseModel):
    tasks: List[BatchFileTask]

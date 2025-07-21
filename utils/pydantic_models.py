from typing import List, Literal
from pydantic import BaseModel


class BatchFileTask(BaseModel):
    file_id: str
    path: str
    status: Literal[
        "pending", # TODO: remove "pending" from the list of statuses
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    ] = "pending"
    result_file_id: str | None = None
    batch_id: str | None = None


class BatchFileTaskList(BaseModel):
    tasks: List[BatchFileTask]

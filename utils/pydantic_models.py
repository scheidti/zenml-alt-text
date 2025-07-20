from typing import List, Literal
from pydantic import BaseModel


class BatchFileTask(BaseModel):
    file_id: str
    path: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result_file_id: str | None = None


class BatchFileTaskList(BaseModel):
    tasks: List[BatchFileTask]
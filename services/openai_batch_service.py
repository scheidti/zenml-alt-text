from uuid import uuid4
from pydantic import Field
from typing import Optional, ClassVar, Tuple, Generator
from openai import OpenAI
from zenml.client import Client
from zenml.logger import get_logger
from zenml.services import (
    BaseService,
    ServiceConfig,
    ServiceStatus,
    ServiceType,
    ServiceState,
)

logger = get_logger(__name__)
client = OpenAI()


class OpenAIBatchConfig(ServiceConfig):
    file_id: str
    name: Optional[str] = None


class OpenAIBatchStatus(ServiceStatus):
    batch_id: Optional[str] = Field(default=None, description="OpenAI batch ID")
    openai_state: Optional[str] = None
    result_file_id: Optional[str] = None


OPENAI_BATCH_SERVICE_TYPE = ServiceType(
    type="external_service",
    flavor="openai_batch",
    name="OpenAI Batch Service",
    description="Tracks the lifecycle of a single OpenAI Batch API job.",
)


class OpenAIBatchService(BaseService):
    SERVICE_TYPE: ClassVar[ServiceType] = OPENAI_BATCH_SERVICE_TYPE
    config: OpenAIBatchConfig
    status: OpenAIBatchStatus

    def provision(self) -> None:
        if self.status.batch_id:
            logger.info(f"OpenAI Batch with ID {self.status.batch_id} already exists.")
            return

        for batch in client.batches.list():
            if batch.input_file_id == self.config.file_id:
                self.status.batch_id = batch.id
                self.status.openai_state = batch.status
                logger.info(f"Found existing OpenAI Batch {batch.id}.")
                return

        job = client.batches.create(
            input_file_id=self.config.file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        self.status.batch_id = job.id
        self.status.openai_state = job.status
        logger.info(f"Created OpenAI Batch with ID {self.status.batch_id}.")

    def deprovision(self, force: bool = False) -> None:
        if not self.status.batch_id:
            logger.info("No OpenAI Batch to cancel.")
            return

        try:
            client.batches.cancel(self.status.batch_id)
            logger.info(f"Canceled OpenAI Batch with ID {self.status.batch_id}.")
        except Exception as e:
            if force:
                logger.error(f"Failed to cancel OpenAI Batch: {e}")
            else:
                raise

    def check_status(self) -> tuple[ServiceState, str]:
        if not self.status.batch_id:
            return ServiceState.INACTIVE, "Not started"

        try:
            job = client.batches.retrieve(self.status.batch_id)
            self.status.openai_state = job.status
            self.status.result_file_id = getattr(job, "output_file_id", None)

            match job.status:
                case "queued" | "validating" | "running" | "finalizing":
                    return ServiceState.ACTIVE, f"{job.status.capitalize()}"
                case "completed":
                    return ServiceState.INACTIVE, "Completed"
                case "failed" | "expired" | "cancelled":
                    return ServiceState.ERROR, f"Batch {job.status}"
                case _:
                    return ServiceState.PENDING_STARTUP, f"State {job.status}"
        except Exception as e:
            return ServiceState.ERROR, str(e)

    def get_logs(
        self, follow: bool = False, tail: Optional[int] = None
    ) -> Generator[str, bool, None]:
        yield "OpenAI Batch API does not provide log output."
        return


def start_openai_batch_service(
    file_id: str,
    name: Optional[str] = None,
) -> OpenAIBatchService:
    config = OpenAIBatchConfig(file_id=file_id, name=name)
    svc = OpenAIBatchService(
        config=config,
        status=OpenAIBatchStatus(),
        uuid=uuid4(),
    )
    svc.start()
    return svc

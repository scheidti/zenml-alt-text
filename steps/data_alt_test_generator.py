import logging

from zenml import step
from datasets import Dataset

logger = logging.getLogger(__name__)


@step
def generate_alt_text(
    data: Dataset, image_row: str = "image", openai_model: str = "gpt-4.1-nano"
) -> Dataset:
    # TODO: Implement the alt text generation logic using OpenAI API
    return data.map(
        lambda x: {"alt_text": f"Generated alt text for {x[image_row]}"},
    )

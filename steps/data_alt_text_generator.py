import json

from pathlib import Path
from zenml import step
from zenml.logger import get_logger
from datasets import Dataset

from utils.images import image_to_base64
from utils.prompts import generate_alt_text_prompt

logger = get_logger(__name__)


@step
def generate_alt_text_batch_files(
    data: Dataset,
    image_row: str = "image",
    openai_model: str = "gpt-4.1-nano",
    batch_size: int = 2000,
    path: str = "./batches",
) -> list[Path]:
    dataset_size = len(data)
    batches = [
        data.select(range(i, min(i + batch_size, dataset_size)))
        for i in range(0, dataset_size, batch_size)
    ]
    files = []
    index = 0
    logger.info(
        f"Generating alt text for {dataset_size} rows in batches of {batch_size}"
    )

    for batch in batches:
        logger.info(f"Processing batch {batches.index(batch) + 1}/{len(batches)}")
        batch_path = Path(path) / f"batch_{batches.index(batch)}.jsonl"
        batch_data = []

        for row in batch:
            entry = {
                "custom_id": f"row_{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": openai_model, "messages": []},
            }
            image = row[image_row]
            base64, format = image_to_base64(image)

            if base64:
                entry["body"]["messages"].append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": generate_alt_text_prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/{format};base64,{base64}",
                            },
                        ],
                    }
                )
                batch_data.append(entry)

            index += 1

        batch_path.write_text("\n".join([json.dumps(item) for item in batch_data]))
        files.append(batch_path)
        logger.info(f"Batch {batches.index(batch)} saved to {batch_path}")

    logger.info(f"Generated {len(files)} batch files in {path}")
    return files

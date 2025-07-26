from dotenv import load_dotenv

from pipelines.data_preparation_pipeline import data_preparation_pipeline
from pipelines.batch_processing_pipeline import batch_processing_pipeline

load_dotenv()


def main() -> None:
    config_path = "configs/data_preparation.yaml"
    data_preparation_pipeline.with_options(config_path=config_path)()
    batch_processing_pipeline.with_options()()


if __name__ == "__main__":
    main()

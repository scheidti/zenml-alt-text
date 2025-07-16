import logging

from pipelines.data_preparation_pipeline import data_preparation_pipeline


def main() -> None:
    config_path = "configs/data_preparation.yaml"
    data_preparation_pipeline.with_options(config_path=config_path)()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    logging.getLogger().setLevel(logging.INFO)
    main()

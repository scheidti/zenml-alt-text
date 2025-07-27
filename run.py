import click
from dotenv import load_dotenv

from pipelines.data_preparation_pipeline import data_preparation_pipeline
from pipelines.batch_processing_pipeline import batch_processing_pipeline
from pipelines.training_pipeline import training_pipeline

load_dotenv()


@click.command()
@click.option(
    "--pipeline",
    type=click.Choice(["data_preparation", "batch_processing", "training"]),
    required=True,
    help="Specify the pipeline to run.",
)
def main(pipeline: str) -> None:
    data_preparation_config_path = "configs/data_preparation.yaml"
    batch_processing_config_path = "configs/batch_processing.yaml"
    training_config_path = "configs/training.yaml"

    match pipeline:
        case "data_preparation":
            data_preparation_pipeline.with_options(
                config_path=data_preparation_config_path
            )()
        case "batch_processing":
            batch_processing_pipeline.with_options(
                config_path=batch_processing_config_path
            )()
        case "training":
            training_pipeline.with_options(config_path=training_config_path)()
        case _:
            click.echo(
                "Invalid pipeline specified. Please choose from 'data_preparation', 'batch_processing', or 'training'."
            )


if __name__ == "__main__":
    main()

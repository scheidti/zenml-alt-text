from zenml import pipeline

from steps.data_loader import load_training_data


@pipeline(name="alt_text_training_pipeline")
def training_pipeline():
    data = load_training_data()
    # TODO: Implement the training pipeline logic

from zenml import pipeline

from steps.data_loader import load_training_data
from steps.training_tasks import train_model


@pipeline(name="alt_text_training_pipeline")
def training_pipeline():
    data = load_training_data()
    train_model(data)

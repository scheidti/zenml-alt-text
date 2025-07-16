from zenml import pipeline
from steps.data_loader import load_data
from steps.data_alt_test_generator import generate_alt_text


@pipeline(name="alt_text_data_preparation_pipeline")
def data_preparation_pipeline():
    data = load_data()
    generate_alt_text(data=data)

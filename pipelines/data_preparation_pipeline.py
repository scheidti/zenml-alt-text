from zenml import pipeline
from steps.data_loader import load_data
from steps.data_alt_text_generator import generate_alt_text_batch_files


@pipeline(name="alt_text_data_preparation_pipeline")
def data_preparation_pipeline():
    data = load_data()
    files = generate_alt_text_batch_files(data=data)
    return files

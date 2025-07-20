from zenml import pipeline

from steps.data_loader import load_data
from steps.data_alt_text_generator import generate_alt_text_batch_files
from steps.data_uploader import upload_files_to_openai


@pipeline(name="alt_text_data_preparation_pipeline")
def data_preparation_pipeline():
    data = load_data()
    files = generate_alt_text_batch_files(data=data)
    batch_tasks = upload_files_to_openai(files=files)
    return batch_tasks

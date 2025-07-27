# ZenML Alt Text Fine-Tuning Project

A machine learning pipeline built with ZenML for fine-tuning LLMs (default Gemma 3n) to generating alternative text descriptions for images.

## 📋 Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
- [ZenML](https://www.zenml.io/) - MLOps for Reliable AI
- [unsloth](https://www.unsloth.ai/) - Easily finetune & train LLMs
- [Hugging Face Account](https://huggingface.co/) - Required for model access (token needed)
- [OpenAI API Key](https://platform.openai.com/api-keys) - Required for OpenAI services

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/scheidti/zenml-alt-text.git
cd zenml-alt-text
```

2. Install uv (if not already installed)

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv sync
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your configuration
```

5. Run pipelines:
```bash
# Data preparation pipeline
python run.py --pipeline data_preparation

# Batch processing pipeline
python run.py --pipeline batch_processing

# Training pipeline
python run.py --pipeline training
```

## ⚙️ Configuration

Each pipeline is configured through YAML files in the `configs/` directory:

- `data_preparation.yaml`: Configuration for data preprocessing
- `batch_processing.yaml`: Configuration for batch inference
- `training.yaml`: Configuration for model training

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
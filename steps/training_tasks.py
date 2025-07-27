import torch
import torch._dynamo as dynamo
from zenml import step
from zenml.logger import get_logger
from datasets import DatasetDict
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from utils.prompts import format_data_for_training

logger = get_logger(__name__)
dynamo.config.cache_size_limit = 2048


@step
def train_model(
    data: DatasetDict,
    model_name: str = "unsloth/gemma-3n-E4B-it",
    hf_repo_id: str = "scheidti/gemma-3n-E4B-it-alt-text-lora",
    hf_merged_repo_id: str = "scheidti/gemma-3n-E4B-it-alt-text-merged-16bit",
):
    train = data["train"]
    validation = data["validation"]

    model, processor = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    converted_train = [format_data_for_training(sample) for sample in train]
    converted_validation = [format_data_for_training(sample) for sample in validation]

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=converted_train,
        eval_dataset=converted_validation,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3,
            warmup_steps=5,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="steps",
            optim="adamw_torch_fused",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
        ),
    )

    trainer_stats = trainer.train()
    logger.info(f"Training completed with stats: {trainer_stats}")
    model.push_to_hub(hf_repo_id)
    processor.push_to_hub(hf_repo_id)
    model.push_to_hub_merged(hf_merged_repo_id, processor, save_method="merged_16bit")
    logger.info("Model and processor pushed to Hugging Face Hub.")

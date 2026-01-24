import os
import sys
import time

import argparse
import gc
import logging

import wandb
import torch

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel, \
    get_peft_model_state_dict
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, \
    EarlyStoppingCallback
from sentence_transformers import SentenceTransformer

from data_prepare import prepare_data_for_finetune
from movie_rec_llm_data_collator import DataCollatorForMovieRec
from movie_rec_llm_trainer import MovieRecLLMTrainer
from loss_calculate_util import compute_trend_similarity, compute_movie_description_similarity, compute_score_similarity

##### Constants and configurations for the fine-tuning process #####
RANDOM_SEED = 1025
FINE_TUNING_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# FINE_TUNING_MODEL_NAME = "meta-llama/Llama-3.1-8B"

CURRENT_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())

WANDB_CONFIG = {
    "entity": "",
    "project": "",
    "name": f"{CURRENT_TIMESTAMP}",
}

HUGGINGFACE_MODEL_LOADING_CONFIG = {
    "max_seq_length": 4096,
    "dtype": torch.bfloat16,
    "device_map": "auto",
    "is_contrastive": True,
    "split_ratio": 0.05
}

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,  # set this to r * 2
    "target_modules": ["q_proj", "v_proj", "o_proj"],  # target modules to apply LoRA, include k_proj, o_proj, etc.
    "lora_dropout": 0.05,  # avoid overfitting
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

QLORA_CONFIG = {
    "r": 16,
    "qlora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],  # target modules to apply LoRA, include k_proj, o_proj, etc.
    "qlora_dropout": 0.1,  # avoid overfitting
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

QLORA_BNB_CONFIG = {
    "load_in_4bit": True,  # load the model in 4-bit quantization
    "bnb_4bit_quant_type": "nf4",  # use the nf4 quantization type
    "bnb_4bit_use_double_quant": True,  # use double quantization
    "bnb_4bit_compute_dtype": torch.float16,  # use float16 for computation
}

CONTRASTIVE_LEARNING_CONFIG = {
    "top_N": 2,  # number of positive pairs
    "DTW": True,  # whether to use DTW for similarity
    "similarity_weight": {
        "score_weight": 0.2,
        "description_weight": 0.4,
        "trend_weight": 0.4,
    },  # weight for the similarity loss, can be adjusted in the future
    "infoNCE_temperature": 0.1,  # temperature for the InfoNCE loss, need to be scheduled in the future
}

LOSS_WEIGHT_CONFIG = {
    "score_pred_loss_weight": 0.5,  # weight for the score prediction loss
    "contrastive_loss_config": 0.5,  # weight for the contrastive learning loss
}

training_config = {
    "output_dir": f"{CURRENT_TIMESTAMP}",
    "per_device_train_batch_size": 8,  # batch size for each GPU, mind that we have 2 GPUs, so total batch size is *4
    "gradient_accumulation_steps": 4,  # gradient accumulation steps, can be adjusted based on the GPU memory
    "learning_rate": 2e-5,  # initial learning rate
    "num_train_epochs": 1,  # number of epochs for training
    "weight_decay": 0.05,  # weight decay for the optimizer
    "lr_scheduler_type": "cosine",  # lr scheduler type, can use "cosine", "polynomial", etc.
    "warmup_ratio": 0.1,  # warmup ratio for the learning rate scheduler
    "save_steps": 150,  # save the model every 150 steps
    "bf16": True,  # use bf16 for training
    "torch_compile": False,  # use torch compile for faster training
    "remove_unused_columns": False,  # do not remove extra columns
    "seed": RANDOM_SEED,  # random seed for reproducibility
    "logging_steps": 10,  # log every 10 steps, related to wandb recording
    "report_to": "wandb",  # report to wandb for experiment tracking
    "evaluation_strategy": "steps",
    "group_by_length": False,
    "ddp_find_unused_parameters": False,
    "dataloader_drop_last": False,
    "save_total_limit": 2,  # how many checkpoints to keep on the disk
    "run_name": WANDB_CONFIG["name"],  # the running name
    "optim": "adamw_torch_fused"
    # optimizer type, see OptimizerNames class in https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_wandb(entity, project, name, config):
    """
    Initialize Weights & Biases (wandb) for experiment tracking, including model hyperparameters, loss curves, and other metrics.
    Args:
        entity (`str`): The entity name for the wandb project.
        project (`str`): The name of the wandb project.
        name (`str`): The name of the run.
        config (`dict`): Other configuration dictionary containing hyperparameters and other settings for the run.
    """

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        resume="allow",
    )

    return run


def run_finetune(
        model_name,
        dataset_path,
        training_config,
        model_loading_config,
        peft_config,
        qlora_bnb_config,
        contrastive_learning_config,
        loss_weight_config,
        dataset_type,
        peft_type="lora",
        resume_from_checkpoint=False,
        lora_adapter_resume_path=""):
    ##### 1. Load the model and the tokenizer #####
    model = None
    tokenizer = None
    peft_model = None
    base_model = None
    if not resume_from_checkpoint:
        ##### 1.1 Dealing with tokenizer #####
        logging.info(f"Loading the model and tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<exp_start>", "<exp_end>"]})

        if peft_type == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=qlora_bnb_config["load_in_4bit"],
                bnb_4bit_use_double_quant=qlora_bnb_config["bnb_4bit_use_double_quant"],
                bnb_4bit_compute_dtype=qlora_bnb_config["bnb_4bit_compute_dtype"],  # H20 can use torch.bfloat16
                bnb_4bit_quant_type=qlora_bnb_config["bnb_4bit_quant_type"],  # usually set to "nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=model_loading_config["dtype"],  # H20 use torch.bfloat1
                device_map=model_loading_config["device_map"],
            )

            # alignment the model
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.use_cache = False

            # If using QLoRA, just do this...
            # If do not open the gradient checkpoint in TrainingArguments, call the code belows
            # model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if peft_type == "lora":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_loading_config["dtype"],  # H20 use torch.bfloat16
                device_map=model_loading_config["device_map"],
                trust_remote_code=True,
            )

            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            model.config.bos_token_id = tokenizer.bos_token_id
            model.config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.bos_token_id = tokenizer.bos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            # model.config.use_cache = False

            logging.info((
                "tok:", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id,
                "| cfg:", model.config.bos_token_id, model.config.eos_token_id, model.config.pad_token_id,
                "| gen:", model.generation_config.bos_token_id, model.generation_config.eos_token_id,
                model.generation_config.pad_token_id
            ))
            # if do not open the gradient checkpoint in TrainingArguments, call the code belows
            model.gradient_checkpointing_enable()
            if getattr(model, "enable_input_require_grads", None):
                model.enable_input_require_grads()

    if resume_from_checkpoint:
        logging.info(f"Start Loading Trained LoRA Adapter Path fron {lora_adapter_resume_path}...")
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_resume_path, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # same as previous fine-tuning
            device_map="auto",
        )
        base_model.gradient_checkpointing_enable()
        if getattr(base_model, "enable_input_require_grads", None):
            base_model.enable_input_require_grads()

        base_model.resize_token_embeddings(len(tokenizer))

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base_model.config.pad_token_id = tokenizer.pad_token_id
        base_model.config.use_cache = False

    ##### 2. Prepare the dataset for fine-tuning #####
    training_set, val_set = prepare_data_for_finetune(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seed=RANDOM_SEED,
        dataset_type=dataset_type,
        is_contrastive=HUGGINGFACE_MODEL_LOADING_CONFIG['is_contrastive'],  # whether to use contrastive learning
        max_length=model_loading_config["max_seq_length"],  # max length of the input sequence
        split_ratio=HUGGINGFACE_MODEL_LOADING_CONFIG['split_ratio']
    )
    logging.info(
        f"Dataset loaded from {dataset_path}. Number of training samples: {len(training_set)}. Number of validation samples: {len(val_set)}")

    ##### 3. Set up training arguments #####
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        max_grad_norm=1,
        warmup_ratio=training_config["warmup_ratio"],
        save_steps=training_config["save_steps"],
        bf16=training_config["bf16"],
        torch_compile=training_config["torch_compile"],
        remove_unused_columns=training_config["remove_unused_columns"],
        seed=training_config["seed"],
        logging_steps=training_config["logging_steps"],
        report_to=training_config["report_to"],
        save_total_limit=training_config["save_total_limit"],
        run_name=training_config["run_name"],
        optim=training_config["optim"],
        gradient_checkpointing=True,  # enable gradient checkpointing to save memory
        label_names=["labels"],
        dataloader_drop_last=training_config["dataloader_drop_last"],
        group_by_length=training_config["group_by_length"],
        logging_first_step=True,
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=training_config['save_steps'],
        load_best_model_at_end=True,
    )

    ##### 4. Set the data collator #####
    data_collator = DataCollatorForMovieRec(
        tokenizer=tokenizer,
        padding=True,  # pad to the maximum length of the sequence
        return_tensors="pt",
        max_length=model_loading_config["max_seq_length"],
    )

    ##### 5. Set up the LoRA configuration #####
    if peft_type == "lora":
        lora_config = LoraConfig(
            r=peft_config["r"],
            lora_alpha=peft_config["lora_alpha"],
            target_modules=peft_config["target_modules"],
            lora_dropout=peft_config["lora_dropout"],
            bias=peft_config["bias"],
            task_type=peft_config["task_type"],
            inference_mode=False,  # set to False for training
        )

    elif peft_type == "qlora":
        lora_config = LoraConfig(
            r=peft_config["r"],
            lora_alpha=peft_config["qlora_alpha"],
            target_modules=peft_config["target_modules"],
            lora_dropout=peft_config["qlora_dropout"],
            bias=peft_config["bias"],
            task_type=peft_config["task_type"],
            inference_mode=False,  # set to False for training
        )

    else:
        logging.error(f"Unsupported PEFT type: {peft_type}. Supported types are 'lora' and 'qlora'.")
        raise ValueError(f"Unsupported PEFT type: {peft_type}. Supported types are 'lora' and 'qlora'.")

    ##### 6. Get the peft model #####
    if resume_from_checkpoint:
        logging.info(f"Resume from {lora_adapter_resume_path}; skip get_peft_model.")
        peft_model = PeftModel.from_pretrained(
            base_model,
            lora_adapter_resume_path,
            is_trainable=True
        )
        peft_model.config.use_cache = False
    else:
        logging.info(f"Initialize PEFT model loaded with {peft_type}.")
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad = False
        peft_model = get_peft_model(model, lora_config)
        peft_model.config.use_cache = False

    peft_model.print_trainable_parameters()

    logging.info((peft_model.active_adapter, peft_model.peft_config[peft_model.active_adapter]))
    trainable = [(n, p.shape) for n, p in peft_model.named_parameters() if p.requires_grad]
    logging.info(f"# of trainable params: {len(trainable)}")
    logging.info(f"trainable params: {trainable}")

    sentence_encoder = SentenceTransformer("all-mpnet-base-v2")
    sentence_encoder.eval()

    trainer = MovieRecLLMTrainer(
        model=peft_model,
        args=training_args,
        encoder=sentence_encoder,
        data_collator=data_collator,
        train_dataset=training_set,
        eval_dataset=val_set,
        contrastive_learning_config=contrastive_learning_config,
        loss_weight_config=loss_weight_config,
        trend_similarity_fn=compute_trend_similarity,
        description_similarity_fn=compute_movie_description_similarity,
        score_similarity_fn=compute_score_similarity,
        tokenizer=tokenizer,
        is_contrastive=HUGGINGFACE_MODEL_LOADING_CONFIG['is_contrastive'],  # whether to use contrastive learning
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    ##### 8. Start the fine-tuning process #####
    logging.info("Starting the fine-tuning process...")
    trainer.train()

    ##### 9. Save the fine-tuned model and tokenizer #####
    logging.info("Saving the fine-tuned model and tokenizer...")
    peft_model.save_pretrained(
        training_config["output_dir"],
        save_embedding_layers=False
    )
    tokenizer.save_pretrained(training_config["output_dir"])
    logging.info("Fine-tuning process completed successfully.")


if __name__ == "__main__":
    ##### 1. Set up the argument parser and arguments #####
    parser = argparse.ArgumentParser(description="Run fine-tuning for the DeepSeek MovieRecLLM model.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="The learning rate of the model to be fine-tuned. Default is 2e-5.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for training per GPU. Default is 8.")
    parser.add_argument("--peft_type", type=str, default="lora", choices=["lora", "qlora"],
                        help="The type of PEFT to use. Default is 'lora'.")
    parser.add_argument("--dataset", type=str, help="Path to the training dataset for fine-tuning.")
    parser.add_argument("--is_contrastive", type=bool, choices=[True, False], default=True,
                        help="Whether to use contrastive learning. Default is True.")
    parser.add_argument("--resume_from_checkpoint", type=bool, choices=[True, False], default=False,
                        help="Whether to resume from the last checkpoint. Default is False.")
    parser.add_argument("--dataset_type", type=str, choices=["douban", "beauty", "baby"], default="douban",
                        help="Type of dataset. Default is douban.")

    args = parser.parse_args()

    training_config["learning_rate"] = args.lr if args.lr else training_config["learning_rate"]
    training_config["per_device_train_batch_size"] = args.batch_size if args.batch_size else training_config[
        "per_device_train_batch_size"]
    finetune_model_name = FINE_TUNING_MODEL_NAME
    dataset_path = f""

    ##### 2. Initialize Weights & Biases (wandb) for experiment tracking #####
    init_wandb(
        WANDB_CONFIG["entity"],
        WANDB_CONFIG["project"],
        WANDB_CONFIG["name"],
        {
            "training_config": training_config,
            "epochs": training_config["num_train_epochs"],
            "dataset": dataset_path,
            "lora_config": LORA_CONFIG if args.peft_type == "lora" else QLORA_CONFIG,
            "contrastive_learning_config": CONTRASTIVE_LEARNING_CONFIG,
            "loss_weight_config": LOSS_WEIGHT_CONFIG,
            "peft_type": args.peft_type,  # use the peft type from the command line argument
            "finetune_model_name": finetune_model_name,
            "dataset_path": dataset_path,
        }
    )

    ##### 3. Run the fine-tuning process #####
    logging.info("Call run_finetune() function to start the finetuning process...")
    run_finetune(
        model_name=finetune_model_name,
        dataset_path=dataset_path,
        training_config=training_config,
        model_loading_config=HUGGINGFACE_MODEL_LOADING_CONFIG,
        peft_config=LORA_CONFIG if args.peft_type == "lora" else QLORA_CONFIG,
        qlora_bnb_config=QLORA_BNB_CONFIG,
        contrastive_learning_config=CONTRASTIVE_LEARNING_CONFIG,
        loss_weight_config=LOSS_WEIGHT_CONFIG,
        dataset_type=args.dataset_type,
        peft_type=args.peft_type,  # use the peft type from the command line argument
        resume_from_checkpoint=False,  # whether to resume from the last checkpoint or not, default is False
        lora_adapter_resume_path="",
    )
    logging.info("Fine-tuning process completed successfully. Check the wandb dashboard for more details.")

    ##### 4. Clean up resources #####
    gc.collect()
    torch.cuda.empty_cache()

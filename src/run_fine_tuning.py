"""
Run fine-tuning training process from this script via command line.

Author: Zhixi Chen
Date: 2025-06-18
"""


import os
import copy
import time

import argparse
import gc
import logging

import wandb
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from src.common.config import load_config, project_path, write_json
from src.common.runtime import environment_record

from .data_prepare import prepare_data_for_finetune
from .movie_rec_llm_data_collator import DataCollatorForMovieRec
from .movie_rec_llm_trainer import MovieRecLLMTrainer
from .loss_calculation_util import compute_trend_similarity, compute_movie_description_similarity, compute_score_similarity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def init_wandb(entity, project, name, config, mode, dir):
    """
    Initialize Weights & Biases (wandb) for experiment tracking, including model hyperparameters, loss curves, and other metrics.
    Args:
        entity (`str`): The entity name for the wandb project.
        project (`str`): The name of the wandb project.
        name (`str`): The name of the run.
        config (`dict`): Other configuration dictionary containing hyperparameters and other settings for the run.
        mode (`str`): The wandb mode, which is `online`, `offline` or `disabled`.
        dir (`str`): The directory where wandb saves its files.
    """

    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        resume="allow",
        mode=mode,
        dir=dir,
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
    sentence_encoder_name,
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
                bnb_4bit_quant_type=qlora_bnb_config["bnb_4bit_quant_type"],        # usually set to "nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=model_loading_config["dtype"], # H20 use torch.bfloat1
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
                torch_dtype=model_loading_config["dtype"], # H20 use torch.bfloat16
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
            torch_dtype=model_loading_config["dtype"],  # same as previous fine-tuning
            device_map=model_loading_config["device_map"],
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
        seed=training_config["seed"],
        dataset_type=dataset_type,
        is_contrastive=model_loading_config['is_contrastive'], # whether to use contrastive learning
        max_length=model_loading_config["max_seq_length"],  # max length of the input sequence
        split_ratio=model_loading_config['split_ratio']
    )
    logging.info(f"Dataset loaded from {dataset_path}. Number of training samples: {len(training_set)}. Number of validation samples: {len(val_set)}")
    
    ##### 3. Set up training arguments #####
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config.get(
            "per_device_eval_batch_size", training_config["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        weight_decay=training_config["weight_decay"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        max_grad_norm=training_config["max_grad_norm"],
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
        gradient_checkpointing=True, # enable gradient checkpointing to save memory
        label_names=["labels"],
        dataloader_drop_last=training_config["dataloader_drop_last"],
        group_by_length=training_config["group_by_length"],
        logging_first_step=True,
        save_strategy="steps",
        eval_strategy=training_config.get("eval_strategy", "no"),
        eval_steps=training_config.get("eval_steps"),
        eval_accumulation_steps=training_config.get("eval_accumulation_steps"),
    )

    ##### 4. Set the data collator #####
    data_collator = DataCollatorForMovieRec(
        tokenizer=tokenizer,
        padding=True, # pad to the maximum length of the sequence
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
            inference_mode=False, # set to False for training
        )
    
    elif peft_type == "qlora":
        lora_config = LoraConfig(
            r=peft_config["r"],
            lora_alpha=peft_config["qlora_alpha"],
            target_modules=peft_config["target_modules"],
            lora_dropout=peft_config["qlora_dropout"],
            bias=peft_config["bias"],
            task_type=peft_config["task_type"],
            inference_mode=False, # set to False for training
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

    sentence_encoder = SentenceTransformer(sentence_encoder_name)
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
        is_contrastive=model_loading_config['is_contrastive'],  # whether to use contrastive learning
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
    return trainer


if __name__ == "__main__":
    ##### 1. Set up the argument parser and arguments #####
    parser = argparse.ArgumentParser(description="Run fine-tuning for the DeepSeek MovieRecLLM model.")
    parser.add_argument("--config", type=str, required=True, help="Path of the JSON config file.")
    parser.add_argument("--run-dir", type=str, required=True, help="The directory of this run. It must not exist yet.")
    parser.add_argument("--device", type=str, help="The device to load the model on. Default is `device_map` in the config.")

    args = parser.parse_args()
    logging.info(f"Set arguments: {vars((args))}")

    ##### 2. Load the constants and configurations for the fine-tuning process #####
    config = load_config(project_path(args.config))
    run_dir = project_path(args.run_dir)
    os.makedirs(run_dir)

    RANDOM_SEED = config["random_seed"]
    FINE_TUNING_MODEL_NAME = config["fine_tuning_model_name"]
    WANDB_CONFIG = config["wandb_config"]
    HUGGINGFACE_MODEL_LOADING_CONFIG = copy.deepcopy(config["huggingface_model_loading_config"])
    LORA_CONFIG = config["lora_config"]
    QLORA_CONFIG = config["qlora_config"]
    QLORA_BNB_CONFIG = copy.deepcopy(config["qlora_bnb_config"])
    CONTRASTIVE_LEARNING_CONFIG = config["contrastive_learning_config"]
    LOSS_WEIGHT_CONFIG = config["loss_weight_config"]
    training_config = copy.deepcopy(config["training_config"])

    # the JSON config stores the names of the torch data types
    HUGGINGFACE_MODEL_LOADING_CONFIG["dtype"] = getattr(torch, HUGGINGFACE_MODEL_LOADING_CONFIG["dtype"])
    QLORA_BNB_CONFIG["bnb_4bit_compute_dtype"] = getattr(torch, QLORA_BNB_CONFIG["bnb_4bit_compute_dtype"])
    if args.device:
        HUGGINGFACE_MODEL_LOADING_CONFIG["device_map"] = {"": args.device}

    wandb_name = WANDB_CONFIG["name"] if WANDB_CONFIG["name"] else f"{run_dir.parent.name}/{run_dir.name}"
    training_config["output_dir"] = str(run_dir / "adapter") # output directory for the fine-tuned model
    training_config["seed"] = RANDOM_SEED # random seed for reproducibility
    training_config["run_name"] = wandb_name # the running name
    finetune_model_name = FINE_TUNING_MODEL_NAME
    dataset_path = str(project_path(config["dataset_path"]))

    ##### 3. Record the configurations and the environment of this run #####
    device = torch.device(args.device if args.device else "cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    results = {
        "model": "TRAIL",
        "status": "running",
        "config": config,
        "seed": RANDOM_SEED,
        "dataset_version": config["dataset_version"],
        "environment": environment_record(device),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "results.json", results)

    ##### 4. Initialize Weights & Biases (wandb) for experiment tracking #####
    init_wandb(
        WANDB_CONFIG["entity"],
        WANDB_CONFIG["project"],
        wandb_name,
        {
            "training_config": training_config,
            "epochs": training_config["num_train_epochs"],
            "dataset": dataset_path,
            "lora_config": LORA_CONFIG if config["peft_type"] == "lora" else QLORA_CONFIG,
            "contrastive_learning_config": CONTRASTIVE_LEARNING_CONFIG,
            "loss_weight_config": LOSS_WEIGHT_CONFIG,
            "peft_type": config["peft_type"],
            "finetune_model_name": finetune_model_name,
            "dataset_path": dataset_path,
        },
        mode=WANDB_CONFIG["mode"],
        dir=str(run_dir),
    )

    ##### 5. Run the fine-tuning process #####
    logging.info("Call run_finetune() function to start the finetuning process...")
    trainer = run_finetune(
        model_name=finetune_model_name,
        dataset_path=dataset_path,
        training_config=training_config,
        model_loading_config=HUGGINGFACE_MODEL_LOADING_CONFIG,
        peft_config=LORA_CONFIG if config["peft_type"] == "lora" else QLORA_CONFIG,
        qlora_bnb_config=QLORA_BNB_CONFIG,
        contrastive_learning_config=CONTRASTIVE_LEARNING_CONFIG,
        loss_weight_config=LOSS_WEIGHT_CONFIG,
        dataset_type=config["dataset"],
        sentence_encoder_name=config["sentence_encoder_name"],
        peft_type=config["peft_type"],
        resume_from_checkpoint=config["resume_from_checkpoint"], # whether to continue from a trained LoRA adapter or not
        lora_adapter_resume_path=str(project_path(config["lora_adapter_resume_path"])),
    )
    logging.info("Fine-tuning process completed successfully. Check the wandb dashboard for more details.")
    wandb.finish()

    ##### 6. Record the results of this run #####
    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    results.update(
        status="complete",
        training_samples=len(trainer.train_dataset),
        validation_samples=len(trainer.eval_dataset),
        validation_history=[
            record for record in trainer.state.log_history if "eval_loss" in record
        ],
        global_steps=trainer.state.global_step,
        adapter_path=training_config["output_dir"],
        elapsed_seconds=time.time() - start_time,
    )
    write_json(run_dir / "results.json", results)

    ##### 7. Clean up resources #####
    gc.collect()
    torch.cuda.empty_cache()

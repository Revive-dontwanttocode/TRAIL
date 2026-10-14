"""
Running model inference in batch mode setting.

Adapted from the author's `run_model_inference_batch.py` (Zhixi Chen, 2025-08-21). Only its `finetune` mode is
kept. Dataset-specific prompts are selected from `prompts.py`.
"""


import json
import logging
import os
import time
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

import torch
import pandas as pd
from tqdm import tqdm

from .prompts import transform_to_input_prompt

from src.common.config import load_config, project_path, write_json
from src.common.runtime import environment_record

from .run_model_evaluation import construct_last_popularity_dataframe, save_inference_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def json_think_tag_info_extraction(response: str, explanation_key: str="explanation_of_score") -> tuple:
    """
    Extract the JSON string from the response. This will utilisse the </think> tag to extract the JSON string.

    Args:
        response (`str`): The response from the model.
        explanation_key (`str`): The JSON key of the explanation.
    Returns:
         `tuple` : A tuple containing the `predict_popularity_score` and `explanation_of_score`. If the JSON string is not found or can not be resolved, return (-1, "N/A").
    """
    search_offset = 0
    think_tag_end = "</think>"
    think_tag_end_pos = response.rfind(think_tag_end)

    if think_tag_end_pos != -1:  # the tag is found
        # Extract the JSON string from the response
        search_offset = think_tag_end_pos + len(think_tag_end)

    json_start_pos = response.find('{', search_offset)
    if json_start_pos == -1:
        return None, None
    predict_popularity_score = None
    explanation_of_score = None
    json_candidate_str = response[json_start_pos:]
    try:
        decoder = json.JSONDecoder()
        json_dict, _ = decoder.raw_decode(json_candidate_str)
        # Judege if the JSON string is valid
        if isinstance(json_dict, dict):
            score_keys = [
                "predict_popularity_score",
                "predict_pop_popularity_score",
                "popularity_score",
                "score"
            ]
            for k in score_keys:
                if k in json_dict:
                    predict_popularity_score = json_dict[k]
                    break
            explanation_of_score = json_dict.get(explanation_key)
            if predict_popularity_score is None or explanation_of_score is None:
                values = list(json_dict.values())
                if len(values) >= 2:
                    predict_popularity_score = predict_popularity_score or values[0]
                    explanation_of_score = explanation_of_score or values[1]
                    logging.warning(
                        f"Key mismatch. Used fallback values: score={predict_popularity_score}, explanation={explanation_of_score}"
                    )
                else:
                    return None, None
            # if one of these two is None type, regarding as failure
            if predict_popularity_score is not None and explanation_of_score is not None:
                # logging.info(f"JSON string ============> : {json_dict}")
                logging.info(f"==========> predict_score: {predict_popularity_score}, explanation: {explanation_of_score}")
                return predict_popularity_score, explanation_of_score
            else:
                logging.warning(
                    "JSON string does not contain both 'predict_popularity_score' and 'explanation_of_score'.")
                return None, None
    except json.JSONDecodeError as e:
        return None, None

    return None, None


def run_finetuned_model_inference_batch_retry(
    inference_mode: str,
    model_name: str, 
    lora_adapter_path: str,
    data_path: str,
    inference_param: dict,
    batch_size: int=32,
    max_retry: int=3,
    device_map="auto",
    is_torch_compile: bool=True,
    max_input_tokens: int=4096,
    item_column: str="item_id",
) -> pd.DataFrame:
    # ##### 1. load the base model and the tokenizer #####
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=True)
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    base_model.resize_token_embeddings(len(tokenizer))

    ##### 2. load the LoRA adapter #####
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
    )
    if is_torch_compile:
        lora_model = torch.compile(lora_model, mode="max-autotune")
    lora_model.eval()

    ##### 2. Load and prepare the model #####
    movie_df_selected = pd.read_csv(data_path, dtype={item_column: str})
    prompts = movie_df_selected.apply(
        lambda row: transform_to_input_prompt(row, tokenizer, inference_mode), axis=1,
    ).tolist()
    explanation_key = "explanation_of_score"
    movie_df_selected["inference_result"] = None
    movie_df_selected["predict_popularity_score"] = None
    movie_df_selected["explanation_of_score"] = None

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc=f"Inference (batch_size={batch_size})"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start: batch_end]
        batch_len = len(batch_prompts)
        batch_results = [None] * batch_len
        
        # record the index which need to inference
        retry_indices = list(range(batch_len))
        retry_prompts = batch_prompts.copy()
        for attempt in range(max_retry):
            if not retry_prompts:
                break
            # the prompts already have the bos token from the chat template
            llm_inputs = tokenizer(
                retry_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                add_special_tokens=False,
            ).to(lora_model.device)
            with torch.inference_mode():
                logging.info(f"Generating outputs for {len(retry_prompts)} prompts...")
                outputs = lora_model.generate(
                    **llm_inputs,
                    max_new_tokens=inference_param['max_new_tokens'],
                    temperature=inference_param['temperature'],
                    top_p=inference_param["top_p"],
                    do_sample=inference_param["do_sample"],
                    top_k=inference_param['top_k'],
                    repetition_penalty=inference_param['repetition_penalty'],
                    num_beams=inference_param["num_beams"],
                )
            # only decode the generated part, the prompt also has `</think>` and JSON-like text
            generated_ids = outputs[:, llm_inputs["input_ids"].shape[1]:]
            batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # see whether we need retry
            next_retry_indices = []
            next_retry_prompts = []
            for j, idx in enumerate(retry_indices):
                response = batch_responses[j]
                batch_results[idx] = (response.strip(), None, None)
                score, explanation = json_think_tag_info_extraction(response=response, explanation_key=explanation_key)
                if score is not None and explanation is not None:
                    batch_results[idx] = (response.strip(), str(score).strip(), str(explanation).strip())
                else:
                    next_retry_indices.append(idx)
                    next_retry_prompts.append(retry_prompts[j])
                    logging.warning(f"Attempt {attempt+1} failed to extract JSON from response. Retrying... Index: {idx}, Response: {response}")
            retry_indices = next_retry_indices
            retry_prompts = next_retry_prompts
        
        batch_indices = list(range(batch_start, batch_end))
        for data_idx, (resp, score, explanation) in zip(batch_indices, batch_results):
            movie_df_selected.at[data_idx, "inference_result"] = resp
            movie_df_selected.at[data_idx, "predict_popularity_score"] = score
            movie_df_selected.at[data_idx, "explanation_of_score"] = explanation

    return movie_df_selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inferencing for the DeepSeek MovieRecLLM model.")
    parser.add_argument("--config", type=str, required=True, help="Path of the JSON config file.")
    parser.add_argument("--run-dir", type=str, required=True, help="The directory of this run. It must not exist yet.")
    parser.add_argument("--device", type=str, help="The device to load the model on. Default is `device_map` in the config.")

    args = parser.parse_args()
    config = load_config(project_path(args.config))
    run_dir = project_path(args.run_dir)
    os.makedirs(run_dir)

    start_time = time.time()
    logging.info("Starting model inference in batch ...")
    INFERENCE_PARAMS = config["inference_params"]

    testing_dataset_path = project_path(config["testing_dataset_path"])
    model_name = config["model_name"]
    base_lora_adapter_path = str(project_path(config["lora_adapter_path"]))
    result_path = run_dir / "inference_result.csv"

    device = torch.device(args.device if args.device else "cuda:0" if torch.cuda.is_available() else "cpu")
    results = {
        "model": "TRAIL",
        "status": "running",
        "config": config,
        "seed": config["seed"],
        "dataset_version": config["dataset_version"],
        "environment": environment_record(device),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "results.json", results)

    # the original script samples without a seed, the seed makes a run repeatable
    set_seed(config["seed"])
    result_df = run_finetuned_model_inference_batch_retry(
        inference_mode=config["dataset"],
        model_name=model_name,
        lora_adapter_path=base_lora_adapter_path,
        data_path=testing_dataset_path,
        inference_param=INFERENCE_PARAMS,
        batch_size=config["batch_size"],
        max_retry=config["max_retry"],
        device_map={"": args.device} if args.device else config["device_map"],
        is_torch_compile=config["torch_compile"],
        max_input_tokens=config["max_input_tokens"],
        item_column=config["item_column"],
    )
    results.update(save_inference_results(result_df, run_dir, config["item_column"], config["evaluation"]))

    # the training-free control: the last observed popularity of every candidate
    last_popularity_df = construct_last_popularity_dataframe(result_df)
    results["last_popularity"] = save_inference_results(last_popularity_df, run_dir / "last_popularity", config["item_column"], config["evaluation"])

    results.update(status="complete", elapsed_seconds=time.time() - start_time)
    write_json(run_dir / "results.json", results)
    logging.info("Inference completed and results saved to {}".format(result_path))

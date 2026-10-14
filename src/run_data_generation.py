"""
Generate training explanations for the dataset selected in the JSON configuration.

Adapted from the author's `run_data_generation.py` (Zhixi Chen, 2025-08-10) and from the
post-processing cells of `amazon_baby_process.ipynb`.
"""


import argparse
import json
import logging
import os
import time
import fcntl
import math
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import pandas as pd
from tqdm import tqdm

from .prompts import transform_to_input_prompt

from src.common.config import load_config, project_path, write_json
from src.common.runtime import environment_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def json_think_tag_info_extraction(response: str) -> tuple:
    """
    Extract the JSON string from the response. This will utilisse the </think> tag to extract the JSON string.
    
    Args:
        response (`str`): The response from the model.
    Returns:
         `tuple` : A tuple containing the `predict_popularity_score` and `explanation_of_score`. If the JSON string is not found or can not be resolved, return (-1, "N/A").
    """
    search_offset = 0
    think_tag_end = "</think>"
    think_tag_end_pos = response.rfind(think_tag_end)
    
    if think_tag_end_pos != -1: # the tag is found
        # Extract the JSON string from the response
        search_offset = think_tag_end_pos + len(think_tag_end)
    
    json_start_pos = response.find('{', search_offset)
    if json_start_pos == -1:
        return None, None
    
    json_candidate_str = response[json_start_pos:]
    try:
        decoder = json.JSONDecoder()
        json_dict, _ = decoder.raw_decode(json_candidate_str)
        # Judege if the JSON string is valid
        if isinstance(json_dict, dict):
            logging.info(f"JSON string ============> : {json_dict}")
            # Extract the predict_popularity_score and explanation_of_score
            predict_popularity_score = json_dict.get("predict_popularity_score")
            explanation_of_score = json_dict.get("explanation_of_score")
            # if one of these two is None type, regarding as failure
            if predict_popularity_score is not None and explanation_of_score is not None:
                return predict_popularity_score, explanation_of_score
            else:
                logging.warning("JSON string does not contain both 'predict_popularity_score' and 'explanation_of_score'.")
                return None, None
    except json.JSONDecodeError as e:
        return None, None

    return None, None


def prepare_vllm_checkpoint(model_name, tokenizer, model_path, local_files_only, dtype=torch.float16):
    """Save the author's vocabulary resize once, before starting the vLLM worker."""
    model_path = Path(model_path)
    if (model_path / "config.json").exists():
        return str(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Preparing resized generation checkpoint at %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"": "cpu"},
        local_files_only=local_files_only,
    )
    model.resize_token_embeddings(len(tokenizer))
    # A partial export must never look like a checkpoint ready for reuse.
    with tempfile.TemporaryDirectory(dir=model_path.parent, prefix="checkpoint-") as temporary:
        staged = Path(temporary) / "model"
        model.save_pretrained(staged)
        tokenizer.save_pretrained(staged)
        staged.rename(model_path)
    del model
    return str(model_path)


def vllm_sampling_kwargs(inference_param):
    """Translate the author's greedy/sampling switch without changing its intent."""
    sampling = inference_param["do_sample"]
    return {
        "max_tokens": inference_param["max_new_tokens"],
        "temperature": inference_param["temperature"] if sampling else 0.0,
        "top_p": inference_param["top_p"] if sampling else 1.0,
        "top_k": (inference_param["top_k"] or -1) if sampling else -1,
        "repetition_penalty": inference_param["repetition_penalty"],
        "skip_special_tokens": True,
    }


def run_model_inference_batch_retry(
    model_name: str, 
    data_path: str,
    result_path: str,
    inference_param: dict,
    dataset_type: str,
    device_map="auto",
    batch_size: int=16,
    max_retry: int=3,
    save_batches: int=10,
    max_input_tokens: int=4096,
    training_set_path=None,
    generation_backend="transformers",
    vllm_config=None,
    local_files_only=False,
    seed=1025,
    legacy_generation=None,
    run_dir=None,
) -> pd.DataFrame:
    samples_df = pd.read_csv(data_path, dtype={"item_id": str})
    if os.path.exists(result_path):
        result_columns = ["inference_result", "predict_popularity_score", "explanation_of_score"]
        saved_df = pd.read_csv(result_path, dtype={"item_id": str, **{column: str for column in result_columns}})
        saved_datasets = set(saved_df["generation_dataset"].dropna())
        if saved_datasets != {dataset_type}:
            raise ValueError("Saved explanations use another dataset prompt; choose a new inference_result_path")
        input_columns = [column for column in samples_df if column != "explanation_of_score"]
        if not saved_df[input_columns].fillna("").astype(str).equals(samples_df[input_columns].fillna("").astype(str)):
            raise ValueError(f"{result_path} does not belong to {data_path}. Use a new result path.")
        samples_df = saved_df
        if "generation_status" not in samples_df:
            samples_df["generation_status"] = samples_df["inference_result"].notna().map({True: "finished", False: "pending"})
    else:
        samples_df["inference_result"] = None
        samples_df["predict_popularity_score"] = None
        samples_df["explanation_of_score"] = None
        samples_df["generation_status"] = "pending"
    samples_df["generation_dataset"] = dataset_type
    # Keep generation metadata outside the training schema.
    for column, key in (("generation_backend", "backend"), ("generation_max_new_tokens", "max_new_tokens")):
        if column not in samples_df:
            samples_df[column] = None
            finished = samples_df["generation_status"] != "pending"
            samples_df.loc[finished, column] = (legacy_generation or {}).get(key)
    data_indices = samples_df.index[samples_df["generation_status"] == "pending"].tolist()
    logging.info("Generation progress: %s / %s rows finished; %s pending.",
                 len(samples_df) - len(data_indices), len(samples_df), len(data_indices))
    if not data_indices:
        save_generation_progress(samples_df, result_path, training_set_path)
        return samples_df

    ##### 1. load the base model and the tokenizer #####
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
    special_tokens = {
        "additional_special_tokens": [
            "<exp_start>", "<exp_end>"
        ]
    }
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens(special_tokens)
    if generation_backend == "transformers":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device_map,
            attn_implementation="sdpa", local_files_only=local_files_only,
        )
        base_model.resize_token_embeddings(len(tokenizer))
    elif generation_backend == "vllm":
        requested = device_map.get("", "cuda:0") if isinstance(device_map, dict) else device_map
        device = torch.device("cuda:0" if requested == "auto" else requested)
        if device.type != "cuda":
            raise ValueError("The vLLM generation backend requires a CUDA device.")
        # vLLM uses logical device zero inside its worker. Respect an existing device mask.
        index = device.index or 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        settings = dict(vllm_config)
        model_path = prepare_vllm_checkpoint(
            model_name, tokenizer, project_path(settings.pop("model_path")), local_files_only,
        )
        from vllm import LLM, SamplingParams
        base_model = LLM(
            model=model_path, dtype="float16", seed=seed,
            max_model_len=max_input_tokens + inference_param["max_new_tokens"],
            generation_config="vllm", **settings,
        )
        sampling_params = SamplingParams(**vllm_sampling_kwargs(inference_param))
    else:
        raise ValueError(f"Unsupported generation backend: {generation_backend}")
    if run_dir is not None:
        actual_device = torch.device("cuda:0") if generation_backend == "vllm" else base_model.device
        write_json(Path(run_dir) / "runtime.json", {
            "generation_backend": generation_backend,
            "dataset": dataset_type,
            "environment": environment_record(actual_device),
            "versions": {name: version(name) for name in
                         (["transformers", "vllm"] if generation_backend == "vllm" else ["transformers"])},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "effective_sampling": vllm_sampling_kwargs(inference_param) if generation_backend == "vllm" else inference_param,
        })
    
    ##### 2. Render only inference inputs; the ground-truth score is never included. #####
    prompts = samples_df.apply(
        lambda row: transform_to_input_prompt(row, tokenizer, dataset_type, "explanation"), axis=1,
    ).tolist()
    for batch_start in tqdm(range(0, len(data_indices), batch_size), desc=f"Inference (batch_size={batch_size})"):
        batch_started = time.monotonic()
        batch_end = min(batch_start + batch_size, len(data_indices))
        batch_indices = data_indices[batch_start: batch_end]
        batch_prompts = [prompts[data_idx] for data_idx in batch_indices]
        batch_len = len(batch_prompts)
        batch_results = [None] * batch_len
        
        # record the index which need to inference
        retry_indices = list(range(batch_len))
        retry_prompts = batch_prompts.copy()
        generated_tokens = 0
        length_limited = 0
        attempts = 0
        for attempt in range(max_retry):
            if not retry_prompts:
                break
            attempts += 1
            # the prompts already have the bos token from the chat template
            if generation_backend == "vllm":
                # Tokenize here: vLLM's truncate_prompt_tokens keeps the opposite end.
                tokens = tokenizer(
                    retry_prompts, truncation=True, max_length=max_input_tokens,
                    add_special_tokens=False, padding=False,
                )
                outputs = base_model.generate(
                    [{"prompt_token_ids": ids} for ids in tokens["input_ids"]],
                    sampling_params=sampling_params, use_tqdm=True,
                )
                generated_ids = [output.outputs[0].token_ids for output in outputs]
                batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_tokens += sum(map(len, generated_ids))
                length_limited += sum(output.outputs[0].finish_reason == "length" for output in outputs)
            else:
                llm_inputs = tokenizer(
                    retry_prompts, return_tensors='pt', padding=True, truncation=True,
                    max_length=max_input_tokens, add_special_tokens=False,
                ).to(base_model.device)
                with torch.no_grad():
                    outputs = base_model.generate(
                        **llm_inputs,
                        max_new_tokens=inference_param['max_new_tokens'],
                        temperature=inference_param['temperature'],
                        top_p=inference_param["top_p"],
                        do_sample=inference_param["do_sample"],
                        top_k=inference_param['top_k'],
                        repetition_penalty=inference_param['repetition_penalty'],
                    )
                # The prompt also contains think tags and JSON-like text.
                generated_ids = outputs[:, llm_inputs["input_ids"].shape[1]:]
                batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # see whether need retry
            next_retry_indices = []
            next_retry_prompts = []
            for j, idx in enumerate(retry_indices):
                response = batch_responses[j]
                score, explanation = json_think_tag_info_extraction(response=response)
                if valid_generated_answer(score, explanation):
                    batch_results[idx] = (response.strip(), str(score).strip(), str(explanation).strip())
                else:
                    # keep the failed response, so the row is not inferenced again when the run continues
                    batch_results[idx] = (response.strip(), None, None)
                    next_retry_indices.append(idx)
                    next_retry_prompts.append(retry_prompts[j])
                    logging.warning(f"Attempt {attempt+1} failed to extract JSON from response. Retrying... Index: {idx}, Response: {response}")
            retry_indices = next_retry_indices
            retry_prompts = next_retry_prompts
        
        for data_idx, (resp, score, explanation) in zip(batch_indices, batch_results):
            samples_df.at[data_idx, "inference_result"] = resp
            samples_df.at[data_idx, "predict_popularity_score"] = score
            samples_df.at[data_idx, "explanation_of_score"] = explanation
            samples_df.at[data_idx, "generation_status"] = "success" if explanation else "failed"
            samples_df.at[data_idx, "generation_backend"] = generation_backend
            samples_df.at[data_idx, "generation_max_new_tokens"] = inference_param["max_new_tokens"]

        if (batch_start // batch_size + 1) % save_batches == 0:
            save_generation_progress(samples_df, result_path, training_set_path)
        elapsed = time.monotonic() - batch_started
        metrics = {
            "backend": generation_backend, "rows": batch_len,
            "accepted": sum(bool(result[2]) for result in batch_results),
            "seconds": elapsed, "rows_per_second": batch_len / elapsed,
            "attempts": attempts,
        }
        if generation_backend == "vllm":
            metrics.update(generated_tokens=generated_tokens, length_limited_attempts=length_limited)
        logging.info("Batch completed: %s", json.dumps(metrics))
        if run_dir is not None:
            with (Path(run_dir) / "generation_batches.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
    save_generation_progress(samples_df, result_path, training_set_path)
    
    return samples_df


def valid_generated_answer(score, explanation):
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(numeric) or numeric < 0 or not isinstance(explanation, str):
        return False
    sections = [explanation.find(marker) for marker in ("[Trend]", "[Feature]", "[Integration]")]
    return 0 <= sections[0] < sections[1] < sections[2]


def save_generation_progress(frame, result_path, training_set_path=None):
    # Atomic replacement leaves the last completed batch readable after interruption.
    result_path = Path(result_path)
    temporary = result_path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(result_path)
    training = construct_training_set(frame)
    if training_set_path is not None:
        training_set_path = Path(training_set_path)
        temporary = training_set_path.with_suffix(".csv.tmp")
        training.to_csv(temporary, index=False)
        temporary.replace(training_set_path)
    completed = int(frame["generation_status"].ne("pending").sum())
    write_json(result_path.parent / "generation_progress.json", {
        "total": len(frame), "completed": completed, "accepted": len(training),
        "failed": completed - len(training), "pending": len(frame) - completed,
        "status": "complete" if completed == len(frame) else "in_progress",
    })
    logging.info("Saved progress: %s / %s finished; %s accepted, %s failed.",
                 completed, len(frame), len(training), completed - len(training))


def construct_training_set(raw_samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the inferenced dataframe into the fine-tuning training set, same as `amazon_baby_process.ipynb`.

    Args:
        raw_samples_df (pd.DataFrame): The dataframe returned by `run_model_inference_batch_retry`.
    Returns:
        pd.DataFrame: The training set with the `explanation_of_score` column.
    """
    # remove columns
    raw_samples_df = raw_samples_df.drop(columns=[
        'inference_result', 'generation_status', 'generation_backend', 'generation_max_new_tokens',
        'generation_dataset',
    ], errors="ignore")

    # deal with explanation, a saved empty explanation is read back as NaN
    raw_samples_df["predict_popularity_score"] = pd.to_numeric(
        raw_samples_df["predict_popularity_score"], errors="coerce"
    )
    raw_samples_df = raw_samples_df.dropna(subset=["predict_popularity_score", "explanation_of_score"]).reset_index(drop=True)
    valid = [valid_generated_answer(score, explanation) for score, explanation in
             zip(raw_samples_df["predict_popularity_score"], raw_samples_df["explanation_of_score"])]
    raw_samples_df = raw_samples_df.loc[valid].copy()

    # explanation_of_score
    raw_samples_df = raw_samples_df[raw_samples_df["explanation_of_score"].str.strip() != ""].reset_index(drop=True)
    raw_samples_df["explanation_of_score"] = raw_samples_df["explanation_of_score"].str.replace('\n', ' ', regex=True)
    raw_samples_df["explanation_of_score"] = (
        raw_samples_df["explanation_of_score"]
        .str.replace('"', '', regex=False)
        .str.replace("'", '', regex=False)
    )

    raw_samples_df = raw_samples_df.drop(columns=["predict_popularity_score"])
    return raw_samples_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TRAIL training explanations using the configured dataset prompt.")
    parser.add_argument("--config", type=str, required=True, help="Path of the JSON config file.")
    parser.add_argument("--device", type=str, help="The device to load the model on. Default is `device_map` in the config.")
    parser.add_argument("--run-dir", type=str, help="Directory for configuration and generation run records.")
    args = parser.parse_args()
    config = load_config(project_path(args.config))

    data_path = project_path(config["data"]["processed_data_path"])
    result_path = project_path(config["data"]["inference_result_path"])
    training_set_path = project_path(config["data"]["training_set_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    training_set_path.parent.mkdir(parents=True, exist_ok=True)
    # Different launcher names must not write concurrently to the same output CSV.
    lock = result_path.with_suffix(".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit(f"Another generation process is using {result_path}")
    run_dir = project_path(args.run_dir) if args.run_dir else training_set_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config["data"]["seed"])
    write_json(run_dir / "config.json", config)
    with (run_dir / "generation_sessions.jsonl").open("a") as stream:
        stream.write(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "config": config}) + "\n")
    start_time = time.time()
    backend = config.get("generation_backend", "transformers")
    if config.get("local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
    logging.info("Starting model inference ...")
    INFERENCE_PARAMS = config["inference_params"]

    model_name = config["model_name"]

    result_df = run_model_inference_batch_retry(
        model_name=model_name,
        data_path=data_path,
        result_path=result_path,
        inference_param=INFERENCE_PARAMS,
        device_map={"": args.device} if args.device else config["device_map"],
        batch_size=config["batch_size"],
        max_retry=config["max_retry"],
        save_batches=config["save_batches"],
        max_input_tokens=config["max_input_tokens"],
        training_set_path=training_set_path,
        generation_backend=backend,
        vllm_config=config.get("vllm"),
        local_files_only=config.get("local_files_only", False),
        seed=config["data"]["seed"],
        legacy_generation=config.get("legacy_generation"),
        run_dir=run_dir,
        dataset_type=config["dataset"],
    )

    logging.info("Inference completed and results saved to {}".format(result_path))

    training_set_df = construct_training_set(result_df)
    logging.info(f"Training set saved to {training_set_path}. {len(training_set_df)} of {len(result_df)} samples have an explanation.")

    configured_device = config["device_map"]
    if isinstance(configured_device, dict):
        configured_device = configured_device[""]
    requested_device = args.device or configured_device
    device = torch.device("cuda:0" if backend == "vllm" else
                          ("cuda:0" if torch.cuda.is_available() else "cpu")
                          if requested_device == "auto" else requested_device)
    record = {
        "model": "TRAIL",
        "status": "complete",
        "task": "TRAIL training explanation generation",
        "dataset": config["dataset"],
        "dataset_version": config["dataset_version"],
        "config": config,
        "seed": config["data"]["seed"],
        "environment": environment_record(device),
        "processed_samples": len(result_df),
        "training_samples": len(training_set_df),
        "elapsed_seconds": time.time() - start_time,
        "generation_backend": backend,
        "generation_backend_counts": result_df["generation_backend"].value_counts().to_dict(),
        "versions": {name: version(name) for name in
                     (["transformers", "vllm"] if backend == "vllm" else ["transformers"])},
    }
    write_json(run_dir / "results.json", record)
    lock.close()

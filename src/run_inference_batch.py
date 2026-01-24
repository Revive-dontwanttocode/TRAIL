import json
import logging
import time
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DOUBAN_SYSTEM_PROMPT = """You are an expert movie popularity analyst. Your task is to predict a movie's popularity score based on its features and historical data, providing a structured explanation.
Your output MUST be a single, valid JSON object with only two keys: "predict_popularity_score" and "explanation_of_score".

### Output Requirements:
1.  `predict_popularity_score`: The predicted movie popularity score at the given <InferenceTimestamp>. This is a non-negative float value, formatted strictly as a string , with no extra text, units, or symbols.
2.  `explanation_of_score`: A text string with three mandatory sections, in order: [Trend]: Analyze <PopularityHistory> using concrete numbers (e.g., recent value, average, peaks/troughs, popularity score percentage change). [Feature]: Refer to at least two specific aspects from <MovieDescription> and explain their influence at <InferenceTimestamp>. [Integration]: Combine [Trend] and [Feature], and justify why the predicted score is reasonable at <InferenceTimestamp>. Use only provided information.

- Output only the required JSON object, with nothing before or after. Do not add explanations, comments, formatting, or any other text outside of the JSON object.

### Input format:
<MovieDescription>...</MovieDescription>
<PopularityHistory>...</PopularityHistory>
<InferenceTimestamp>...</InferenceTimestamp>
"""

AMAZON_BEAUTY_SYSTEM_PROMPT = """You are an expert product popularity analyst. Your task is to predict a product's popularity score based on its product description and historical popularity trends, and provide a structured explanation.
Your output MUST be a single, valid JSON object with only two keys: "predict_popularity_score" and "explanation_of_score".

### Output Requirements:
1. `predict_popularity_score`: The predicted product popularity score at the given <PredictTimestamp>. It must be a non-negative float, formatted strictly as a string (no units, no extra text).
2. `explanation_of_score`: A text string with three mandatory sections, in order:
    - [Trend]: Analyze the historical popularity trend in `<HistoricalPopularityTrends></HistoricalPopularityTrends>` tag using concrete numbers (e.g., recent value, average, peaks/troughs, percentage change).
    - [Feature]: Refer to at least two specific aspects in `<ProductDescription></ProductDescription>` tag (e.g., brand, category, price, awards, functionality, design) and explain their influence at <PredictTimestamp>.
    - [Integration]: Combine [Trend] and [Feature] into a coherent justification of why the predicted score is reasonable at <PredictTimestamp>.

### Reasoning Format:
- Wrap your step-by-step reasoning in <think>...</think> tags.
- Do not include <think> in the final output. 

### Output Format:
- Output must be a single, valid JSON object with only two keys: "predict_popularity_score" and "explanation_of_score".
- No extra text before or after the JSON.

### Input format:
<ProductDescription>...</ProductDescription>  
<HistoricalPopularityTrends>...</HistoricalPopularityTrends>  
<PredictTimestamp>...</PredictTimestamp>
Please provide only the reasoning steps followed by the final JSON output in the format {{"predict_popularity_score": "...", "explanation_of_score": "..."}}.
"""


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
            explanation_of_score = json_dict.get("explanation_of_score")
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
                logging.info(
                    f"==========> predict_score: {predict_popularity_score}, explanation: {explanation_of_score}")
                return predict_popularity_score, explanation_of_score
            else:
                logging.warning(
                    "JSON string does not contain both 'predict_popularity_score' and 'explanation_of_score'.")
                return None, None
    except json.JSONDecodeError as e:
        return None, None

    return None, None


def transform_movie_to_input_prompt(row: pd.Series, tokenizer: AutoTokenizer, system_prompt: str) -> str:
    """
    Transform the processed dataset into one input prompt.

    Args:
        row (pd.Series): A row of the processed dataset.
    Returns:
        str: The input prompt for the model.
    """

    # Extract the relavant information from the row
    movie_description = row.get("movie_description")
    popularity_history = row.get("historical_popularity_trends")
    inference_timestamp = row.get("predict_timestamp")

    # Formatting the prompt
    user_prompt_formatted = (
        f"### Input:\n"
        f"<MovieDescription>{movie_description}</MovieDescription>\n"
        f"<PopularityHistory>{popularity_history}</PopularityHistory>\n"
        f"<InferenceTimestamp>{inference_timestamp}</InferenceTimestamp>"
    )

    final_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_formatted},
    ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return final_prompt


def transform_amazon_beauty_to_input_prompt(row: pd.Series, tokenizer: AutoTokenizer, system_prompt: str) -> str:
    product_description = row.get("product_description")
    popularity_history = row.get("historical_popularity_trends")
    predict_timestamp = row.get("predict_timestamp")

    # Formatting the prompt
    user_prompt_formatted = (
        f"### Input:\n"
        f"<ProductDescription>{product_description}</ProductDescription>\n"
        f"<HistoricalPopularityTrends>{popularity_history}</HistoricalPopularityTrends>\n"
        f"<PredictTimestamp>{predict_timestamp}</PredictTimestamp>"
    )
    final_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_formatted},
    ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return final_prompt


def transform_gt_movie_to_input_prompt(row: pd.Series, tokenizer: AutoTokenizer, system_prompt: str) -> str:
    """
    Transform the processed dataset into one input prompt.

    Args:
        row (pd.Series): A row of the processed dataset.
    Returns:
        str: The input prompt for the model.
    """

    # Extract the relevant information from the row
    movie_description = row.get("movie_description")
    popularity_history = row.get("historical_popularity_trends")
    inference_timestamp = row.get("predict_timestamp")
    ground_truth_popularity_score = row.get("ground_truth_popularity_score")

    # Formatting the prompt
    user_prompt_formatted = (
        f"### Input:\n"
        f"<GivenPopularityScore>{ground_truth_popularity_score}</GivenPopularityScore>\n"
        f"<MovieDescription>{movie_description}</MovieDescription>\n"
        f"<PopularityHistory>{popularity_history}</PopularityHistory>\n"
        f"<InferenceTimestamp>{inference_timestamp}</InferenceTimestamp>"
    )

    final_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_formatted},
    ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return final_prompt


def transform_gt_amazon_to_input_prompt(row: pd.Series, tokenizer: AutoTokenizer, system_prompt: str) -> str:
    """
    Transform the processed dataset into one input prompt.

    Args:
        row (pd.Series): A row of the processed dataset.
    Returns:
        str: The input prompt for the model.
    """

    # Extract the relavant information from the row
    product_description = row.get("product_description")
    popularity_history = row.get("historical_popularity_trends")
    inference_timestamp = row.get("predict_timestamp")
    ground_truth_popularity_score = row.get("ground_truth_popularity_score")

    # Formatting the prompt
    user_prompt_formatted = (
        f"### Input:\n"
        f"<GivenPopularityScore>{ground_truth_popularity_score}</GivenPopularityScore>\n"
        f"<ProductDescription>{product_description}</ProductDescription>\n"
        f"<PopularityHistory>{popularity_history}</PopularityHistory>\n"
        f"<InferenceTimestamp>{inference_timestamp}</InferenceTimestamp>"
    )

    final_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_formatted},
    ],
        tokenize=False,
        add_generation_prompt=True,
    )

    return final_prompt


def run_finetuned_model_inference_batch_retry(
        inference_mode: str,
        model_name: str,
        lora_adapter_path: str,
        data_path: str,
        inference_param: dict,
        batch_size: int = 32,
        max_retry: int = 3,
) -> pd.DataFrame:
    # ##### 1. load the base model and the tokenizer #####
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base_model.resize_token_embeddings(len(tokenizer))

    ##### 2. load the LoRA adapter #####
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
    )
    lora_model = torch.compile(lora_model, mode="max-autotune")
    lora_model.eval()

    ##### 2. Load and prepare the model #####
    movie_df_selected = pd.read_csv(data_path)
    prompts = None
    if inference_mode == "douban":
        prompts = movie_df_selected.apply(
            lambda row: transform_movie_to_input_prompt(row, tokenizer, DOUBAN_SYSTEM_PROMPT), axis=1).tolist()
    if inference_mode == "beauty" or inference_mode == "baby":
        prompts = movie_df_selected.apply(
            lambda row: transform_amazon_beauty_to_input_prompt(row, tokenizer, AMAZON_BEAUTY_SYSTEM_PROMPT),
            axis=1).tolist()
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
            llm_inputs = tokenizer(
                retry_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
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
                )
            batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # see whether we need retry
            next_retry_indices = []
            next_retry_prompts = []
            for j, idx in enumerate(retry_indices):
                response = batch_responses[j]
                score, explanation = json_think_tag_info_extraction(response=response)
                if score is not None and explanation is not None:
                    batch_results[idx] = (response.strip(), str(score).strip(), str(explanation).strip())
                else:
                    next_retry_indices.append(idx)
                    next_retry_prompts.append(retry_prompts[j])
                    logging.warning(
                        f"Attempt {attempt + 1} failed to extract JSON from response. Retrying... Index: {idx}, Response: {response}")
            retry_indices = next_retry_indices
            retry_prompts = next_retry_prompts

        for idx in retry_indices:
            batch_results[idx] = ("", None, None)

        batch_indices = list(range(batch_start, batch_end))
        for data_idx, (resp, score, explanation) in zip(batch_indices, batch_results):
            movie_df_selected.at[data_idx, "inference_result"] = resp
            movie_df_selected.at[data_idx, "predict_popularity_score"] = score
            movie_df_selected.at[data_idx, "explanation_of_score"] = explanation

    return movie_df_selected


def fill_missing_inference_rows(
        inference_mode: str,
        lora_adapter_path: str,
        movie_df_selected: pd.DataFrame,
        inference_param: dict,
        batch_size: int = 16,
        max_retry: int = 3,
) -> pd.DataFrame:

    def _is_missing(x):
        return (x is None) or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == "")

    movie_df_selected.loc[movie_df_selected["item_id"].isin([10180, 9659]), "predict_popularity_score"] = np.nan

    logging.info('Finding missing inference_results...')
    need_mask = (movie_df_selected["predict_popularity_score"].apply(_is_missing) | movie_df_selected[
        "explanation_of_score"].apply(_is_missing)) & (movie_df_selected["ground_truth_popularity_score"] > 5)

    indices_to_fill = movie_df_selected.index[need_mask].tolist()
    if len(indices_to_fill) == 0:
        logging.info("No missing rows to fill. Nothing to do.")
        return movie_df_selected

    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.resize_token_embeddings(len(tokenizer))
    lora_model = PeftModel.from_pretrained(
        base,
        lora_adapter_path,
        is_trainable=False
    ).eval()

    def _build_prompt(row):
        if inference_mode == "douban":
            return transform_movie_to_input_prompt(row, tokenizer, DOUBAN_SYSTEM_PROMPT)
        elif inference_mode in ("beauty", "baby"):
            return transform_amazon_beauty_to_input_prompt(row, tokenizer, AMAZON_BEAUTY_SYSTEM_PROMPT)
        else:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")

    prompts = []
    for idx in indices_to_fill:
        row = movie_df_selected.loc[idx]
        prompts.append(_build_prompt(row))

    for batch_start in tqdm(range(0, len(prompts), batch_size),
                            desc=f"Fill missing inference (batch_size={batch_size})"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start: batch_end]
        batch_indices = indices_to_fill[batch_start: batch_end]
        batch_len = len(batch_prompts)

        batch_results = [None] * batch_len
        retry_positions = list(range(batch_len))
        retry_prompts = batch_prompts.copy()

        for attempt in range(max_retry):
            if not retry_prompts:
                break

            llm_inputs = tokenizer(
                retry_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(lora_model.device)

            with torch.no_grad():
                outputs = lora_model.generate(
                    **llm_inputs,
                    max_new_tokens=inference_param['max_new_tokens'],
                    temperature=inference_param['temperature'],
                    top_p=inference_param["top_p"],
                    do_sample=inference_param["do_sample"],
                    top_k=inference_param['top_k'],
                    repetition_penalty=inference_param['repetition_penalty'],
                )

            batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            next_retry_positions = []
            next_retry_prompts = []

            for j, pos in enumerate(retry_positions):
                response = batch_responses[j]
                score, explanation = json_think_tag_info_extraction(response=response)
                if score is not None and explanation is not None:
                    batch_results[pos] = (response.strip(), str(score).strip(), str(explanation).strip())
                else:
                    next_retry_positions.append(pos)
                    next_retry_prompts.append(retry_prompts[j])
                    logging.warning(
                        f"[fill-missing] Attempt {attempt + 1} failed to extract JSON. "
                        f"GlobalIdx={batch_indices[pos]} Response: {response}..."
                    )

            retry_positions = next_retry_positions
            retry_prompts = next_retry_prompts

        for pos in retry_positions:
            batch_results[pos] = ("", None, None)

        for local_pos, global_idx in enumerate(batch_indices):
            resp, score, explanation = batch_results[local_pos]
            movie_df_selected.at[global_idx, "inference_result"] = resp
            movie_df_selected.at[global_idx, "predict_popularity_score"] = score
            movie_df_selected.at[global_idx, "explanation_of_score"] = explanation

    return movie_df_selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inferencing for the DeepSeek MovieRecLLM model.")
    parser.add_argument("--lora_path", type=str, help="The lora adapter path of the inferencing model.")
    parser.add_argument("--mode", type=str, choices=["finetune", "finetune_merged", "raw", "exp", "fill"],
                        help="The mode to run the model. Choose from 'finetune' or 'raw'.")
    parser.add_argument("--max_new_tokens", type=int, help="The maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, help="The temperature for the model inference.")
    parser.add_argument("--top_p", type=float, help="The top-p sampling for the model inference.")
    parser.add_argument("--top_k", type=int, help="The top-k sampling for the model inference.")
    parser.add_argument("--num_beams", type=int, help="The number of beam search.")
    parser.add_argument("--model", type=str, help="The LLM we choose.")
    parser.add_argument("--testing_set_type", type=str, choices=["douban", "beauty", "baby"],
                        help="The testing set we choose.")
    parser.add_argument("--merged_model_base", type=str, help="The base model for the merged model.")
    parser.add_argument("--save_filename", type=str, help="The data saving path.")

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.info("Starting model inference in batch ...")
    INFERENCE_PARAMS = {
        "max_new_tokens": 4096,
        "temperature": 0.35,
        "do_sample": True,
        "top_p": args.top_p if args.top_p else 0.95,
        "top_k": args.top_k if args.top_k else 35,
        "repetition_penalty": 1.2,
        "num_beams": 1
    }

    testing_dataset_path = ''
    if args.testing_set_type == "douban":
        testing_dataset_path = ""

    if args.testing_set_type == "beauty":
        testing_dataset_path = ""

    if args.testing_set_type == "baby":
        testing_dataset_path = ""

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # need to change this for different dataset
    base_lora_adapter_path = ""
    result_path = f""

    if args.mode == "finetune":
        result_df = run_finetuned_model_inference_batch_retry(
            inference_mode=args.testing_set_type,
            model_name=model_name,
            lora_adapter_path=base_lora_adapter_path,
            data_path=testing_dataset_path,
            inference_param=INFERENCE_PARAMS,
            batch_size=32,
            max_retry=1,
        )
        result_df.to_csv(result_path, index=False)


    if args.mode == "fill":
        result_df = fill_missing_inference_rows(
            inference_mode=args.testing_set_type,
            lora_adapter_path=base_lora_adapter_path,
            movie_df_selected=pd.read_csv(""),
            inference_param=INFERENCE_PARAMS,
            batch_size=16,
            max_retry=1
        )
        result_df.to_csv(result_path, index=False)

    logging.info("Inference completed and results saved to {}".format(result_path))

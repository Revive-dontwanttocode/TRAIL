from transformers import AutoTokenizer
import dirtyjson
import json

DOUBAN_SYSTEM_PROMPT = """You are a movie popularity analyst. Your task is to predict a movie's popularity score based on its features and historical data, providing a structured explanation.
Your output MUST be a single, valid JSON object with only two keys: "predict_popularity_score" and "explanation_of_score".

### Output Requirements:
1.  `predict_popularity_score`: The predicted movie popularity score at the given <InferenceTimestamp>. This is a non-negative float value, formatted strictly as a string , with no extra text, units, or symbols.
2.  `explanation_of_score`: A text string with three mandatory sections, in order:
    - [Trend]: Analyze <PopularityHistory> using concrete numbers (e.g., recent value, average, peaks/troughs, popularity score percentage change).
    - [Feature]: Refer to at least two specific aspects from <MovieDescription> and explain their influence at <InferenceTimestamp>.
    - [Integration]: Combine [Trend] and [Feature], and justify why the predicted score is reasonable at <InferenceTimestamp>. Use only provided information.

- Output only the JSON object, with nothing before or after.

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
"""

AMAZON_BABY_SYSTEM_PROMPT = """You are an expert product popularity analyst. Your task is to predict a product's popularity score based on its product description and historical popularity trends, and provide a structured explanation.
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
"""

ANSWER_TEMPLATE = """{{"predict_popularity_score": "{}", "explanation_of_score": "{}"}}"""


def find_subsequence(source, target):
    if isinstance(target, int):
        target = [target]
    for i in range(len(source) - len(target) + 1):
        if source[i:i + len(target)] == target:
            return i
    return -1


def resolve_trend(trend_str: str) -> list:
    """
    Given a json string of trend values, return a list of trend values.

    Args:
        trend_str (`str`): A json string of trend values.

    Returns:
        list: A list of trend values.
    """
    # Parse the json string
    trend_json_result = dirtyjson.loads(trend_str, search_for_first_object=True)
    # Extract the trend values and save to a list
    trend_values = [float(record["popularity_score"]) for record in trend_json_result]
    # Return the trend values
    return trend_values


def tokenize_data_sample(sample: dict, tokenizer: AutoTokenizer, max_length, dataset_type, is_contrastive=True) -> dict:
    ##### 1. Prepare the prompt and full input text #####
    if dataset_type == "douban":
        user_input = (
            f"### Input:\n"
            f"<MovieDescription>{sample['movie_description']}</MovieDescription>\n"
            f"<PopularityHistory>{sample['historical_popularity_trends']}</PopularityHistory>\n"
            f"<InferenceTimestamp>{sample['predict_timestamp']}</InferenceTimestamp>"
        )
        answer = json.dumps(
            {
                "predict_popularity_score": str(sample["ground_truth_popularity_score"]),
                "explanation_of_score": "<exp_start> " + str(sample["explanation_of_score"]) + "<exp_end>"
            }
        )
        prompt_only_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DOUBAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": ""},
            ],
            tokenize=False,  # retrieve the text itself first
            add_generation_prompt=False
        )
        prompt_full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DOUBAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    if dataset_type == "beauty":
        user_input = (
            f"### Input:\n"
            f"<ProductDescription>{sample['movie_description']}</ProductDescription>\n"
            f"<PopularityHistory>{sample['historical_popularity_trends']}</PopularityHistory>\n"
            f"<InferenceTimestamp>{sample['predict_timestamp']}</InferenceTimestamp>"
        )
        answer = json.dumps(
            {
                "predict_popularity_score": str(sample["ground_truth_popularity_score"]),
                "explanation_of_score": "<exp_start> " + str(sample["explanation_of_score"]) + "<exp_end>"
            }
        )
        prompt_only_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": AMAZON_BEAUTY_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": ""},
            ],
            tokenize=False,  # retrieve the text itself first
            add_generation_prompt=False
        )
        prompt_full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": AMAZON_BEAUTY_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    if dataset_type == "baby":
        user_input = (
            f"### Input:\n"
            f"<ProductDescription>{sample['movie_description']}</ProductDescription>\n"
            f"<PopularityHistory>{sample['historical_popularity_trends']}</PopularityHistory>\n"
            f"<InferenceTimestamp>{sample['predict_timestamp']}</InferenceTimestamp>"
        )
        answer = json.dumps(
            {
                "predict_popularity_score": str(sample["ground_truth_popularity_score"]),
                "explanation_of_score": "<exp_start> " + str(sample["explanation_of_score"]) + "<exp_end>"
            }
        )
        prompt_only_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": AMAZON_BABY_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": ""},
            ],
            tokenize=False,  # retrieve the text itself first
            add_generation_prompt=False
        )
        prompt_full_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": AMAZON_BABY_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

    ##### 2. Use the same tokenizer to encode prompt_only_text and prompt_full_text #####
    prompt_only_text_ids = tokenizer(
        prompt_only_text,
        padding=False,
        max_length=max_length,
        return_tensors=None,
        truncation=True,
        add_special_tokens=False,
    )

    model_inputs = tokenizer(
        prompt_full_text,
        padding=False,
        max_length=max_length,
        return_tensors=None,
        truncation=True,
        add_special_tokens=False
    )

    ##### 3. Deal with input_ids and write it back to model_inputs and calculate prompt_len #####
    input_ids = model_inputs["input_ids"]
    if isinstance(input_ids, list) and len(input_ids) == 1 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    model_inputs["input_ids"] = input_ids

    prompt_only_text_ids = prompt_only_text_ids["input_ids"]
    prompt_only_text_len = len(prompt_only_text_ids)

    ##### 4. Locate the special tokens #####
    exp_start_id = tokenizer.convert_tokens_to_ids("<exp_start>")
    exp_end_id = tokenizer.convert_tokens_to_ids("<exp_end>")
    if (exp_start_id not in input_ids) or (exp_end_id not in input_ids):
        return None

    exp_start_index = find_subsequence(input_ids, exp_start_id)
    exp_end_index = find_subsequence(input_ids, exp_end_id)
    if exp_start_index == -1 or exp_end_index == -1:
        return None  # skip abnormal data

    # Add other information to the model_inputs
    model_inputs["ground_truth_popularity_score"] = float(sample["ground_truth_popularity_score"])
    # model_inputs["movieid"] = str(sample["movieid"])
    model_inputs["historical_popularity_trends"] = sample["historical_popularity_trends"]
    model_inputs["predict_timestamp"] = sample["predict_timestamp"]
    model_inputs["movie_description"] = sample["movie_description"]
    model_inputs["list_historical_popularity_trends"] = resolve_trend(sample["historical_popularity_trends"])

    # construct labels
    labels = model_inputs["input_ids"][:]  # Clone the input_ids to create labels
    num_mask = min(prompt_only_text_len - 1, len(labels))
    # Mask the prompt part in labels
    for i in range(num_mask):
        labels[i] = -100  # -100 is the label for padding tokens in Hugging Face transformers

    # Mark other labels that between <exp_start> and <exp_end>. If you want to add the explanation part into the supervised learning, just annotate this part
    for i in range(exp_start_index, min(exp_end_index + 1, len(labels))):
        labels[i] = -100

    labels[exp_start_index] = -100
    labels[exp_end_index] = -100

    # add labels to model_inputs
    model_inputs["labels"] = labels
    return model_inputs

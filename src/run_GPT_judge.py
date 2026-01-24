CHATGPT_API_KEY = ''

SYSTEM_PROMPT = """
### Instruction:
Your goal is to strictly and objectively assess the explanation's quality. Your evaluations must clearly distinguish between strong, average, and poor explanations.
You will be given the true historical trend data, item description and metadata, the prediction timestamp, the ground-truth popularity score, the model’s predicted popularity score, and its generated explanation.
Please rate the explanation from 4 perspectives (scores between 0 and 1), and give a short comment: 

1. Truthfulness of historical trend
Look at how the explanation describes the historical trend — does it accurately reflect the real trend’s direction, peaks, and fluctuations? Give a score between 0 and 1 for how truthful it is with respect to the historical trend, where 1 means completely accurate” and 0 means completely inconsistent.

2. Truthfulness of item metadata
Look at the metadata — are the mentioned product attributes (such as item name, item special feature, etc.) actually present well in the provided item description? Give a score between 0 and 1 for how truthful it is with respect to the metadata, where 1 means completely accurate” and 0 means completely inconsistent.” 

3. Predictiveness / Consistency
Judge whether the explanation's reasoning about future popularity makes sense given the past trend. If the trend shows steady growth and the explanation predicts continued or strong demand, that’s consistent and should get a high score. If the explanation predicts something that clearly contradicts the trend without any justification, that’s inconsistent and should get a low score.
Rate it from 0 to 1, where 1 means highly consistent and reasonable. 

4. Logic & Language Quality
Give a score from 0 to 1 based on how logical, consistent, and well-written the explanation is.
Do not judge formatting or section structure; focus only on clarity, coherence, and internal logic. 

Return your output in JSON like:

{
  "trend_truthfulness_score": "...", 
  "metadata_truthfulness_score": "...", 
  "predictiveness_score": "...",
  "logic_score": "...",
  "comment": "…"
}"""

INPUT_PROMPT = """### Input:
<ItemDescription> {item_description} </ItemDescription>
<HistoricalPopularityTrend> {historical_trend} </HistoricalPopularityTrend>
<PredictedTimestamp> {predicted_timestamp} </PredictedTimestamp>
<TruePopularityScore> {true_popularity_score} </TruePopularityScore>
<PredictPopularityScore> {predict_popularity_score} </PredictPopularityScore>
<GeneratedExplanation> {generated_explanation} </GeneratedExplanation>
"""

import argparse
from openai import OpenAI
import logging
import pandas as pd
import json

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def ask_gpt_for_answer(client, system_prompt: str, prompt: str, model: str = "gpt-4.1-mini",
                       temperature: int = 0.3) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def evaluate_one_explanation(row: pd.Series, client: OpenAI, mode: str) -> dict:
    ##### 1. Fill the prompt template #####
    prompt = None
    # chech whether the row has N/A values
    if row.isnull().any():
        logging.warning(f"Row has N/A values, skipping evaluation: {row.to_dict()}")
        return {
            "trend_truthfulness_score": 0,
            "metadata_truthfulness_score": 0,
            "predictiveness_score": 0,
            "logic_score": 0,
            "comment": ""
        }

    if mode == "beauty" or mode == "baby" or mode == 'baby_baseline':
        prompt = INPUT_PROMPT.format(
            item_description=row["product_description"],
            historical_trend=row["historical_popularity_trends"],
            predicted_timestamp='2014-06',
            true_popularity_score=row["ground_truth_popularity_score"],
            predict_popularity_score=row["predict_popularity_score"],
            generated_explanation=row["explanation_of_score"]
        )

    if mode == "douban" or mode == 'douban_baseline':
        prompt = INPUT_PROMPT.format(
            item_description=row["movie_description"],
            historical_trend=row["historical_popularity_trends"],
            predicted_timestamp='2019-11',
            true_popularity_score=row["ground_truth_popularity_score"],
            predict_popularity_score=row["predict_popularity_score"],
            generated_explanation=row["explanation_of_score"]
        )

    ##### 2. Call the LLM API to get the evaluation result #####
    response_text = ask_gpt_for_answer(client, SYSTEM_PROMPT, prompt)
    logging.info(f"Evaluation response text: {response_text}")
    ##### 3. Parse the JSON result and return #####
    try:
        evaluate_data = json.loads(response_text)
        evaluate_dict = {
            "trend_truthfulness_score": float(evaluate_data.get("trend_truthfulness_score", 0)),
            "metadata_truthfulness_score": float(evaluate_data.get("metadata_truthfulness_score", 0)),
            "predictiveness_score": float(evaluate_data.get("predictiveness_score", 0)),
            "logic_score": float(evaluate_data.get("logic_score", 0)),
            "comment": evaluate_data.get("comment", "")
        }
        return evaluate_dict
    except json.JSONDecodeError:
        logging.error(f"Failed to parse JSON response: {response_text}")
        return {
            "trend_truthfulness_score": 0,
            "metadata_truthfulness_score": 0,
            "predictiveness_score": 0,
            "logic_score": 0,
            "comment": ""
        }


def explanation_score_evaluation(inference_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    client = OpenAI(api_key=CHATGPT_API_KEY)

    inference_df = inference_df.copy()
    inference_df['trend_truthfulness_score'] = 0.0
    inference_df['metadata_truthfulness_score'] = 0.0
    inference_df['predictiveness_score'] = 0.0
    inference_df['logic_score'] = 0.0
    inference_df['comment'] = ""

    for idx, row in tqdm(inference_df.iterrows(), total=len(inference_df)):
        eval_result = evaluate_one_explanation(row, client, mode)

        inference_df.at[idx, "trend_truthfulness_score"] = eval_result.get("trend_truthfulness_score")
        inference_df.at[idx, "metadata_truthfulness_score"] = eval_result.get("metadata_truthfulness_score")
        inference_df.at[idx, "predictiveness_score"] = eval_result.get("predictiveness_score")
        inference_df.at[idx, "logic_score"] = eval_result.get("logic_score")
        inference_df.at[idx, "comment"] = eval_result.get("comment")

    return inference_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path", type=str, required=True, help="Path to the input CSV file with explanations to evaluate.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output CSV file with evaluation scores.")
    args = parser.parse_args()

    logging.info(f"Loading data ...")
    inference_df = pd.read_csv(f"")

    inference_df = inference_df.dropna(subset=["predict_popularity_score"]).reset_index(drop=True)
    logging.info("Starting explanation score evaluation...")
    evaluated_df = explanation_score_evaluation(inference_df, args.mode)

    logging.info(f"Saving evaluated data to {args.output_path}")
    evaluated_df.to_csv(args.output_path, index=False)

    logging.info("Evaluation completed.")

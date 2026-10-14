"""Score saved explanations with the author's optional GPT judge prompt."""

import argparse
import json

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from .common.config import load_config, project_path, write_json
from .prompts import EXPLANATION_JUDGE_SYSTEM_PROMPT, EXPLANATION_JUDGE_USER_TEMPLATE


SCORE_FIELDS = (
    "trend_truthfulness_score", "metadata_truthfulness_score",
    "predictiveness_score", "logic_score",
)


def evaluate_explanations(frame, config):
    client = OpenAI()
    frame = frame.copy()
    for field in SCORE_FIELDS:
        frame[field] = float("nan")
    frame["comment"] = ""
    frame["judge_response"] = ""
    frame["judge_status"] = "missing_prediction"
    ready = frame.dropna(subset=["predict_popularity_score", "explanation_of_score"])
    for index, row in tqdm(ready.iterrows(), total=len(ready), desc="Judging explanations"):
        prompt = EXPLANATION_JUDGE_USER_TEMPLATE.format(
            item_description=row["item_description"],
            historical_trend=row["historical_popularity_trends"],
            predicted_timestamp=row["predict_timestamp"],
            true_popularity_score=row["ground_truth_popularity_score"],
            predict_popularity_score=row["predict_popularity_score"],
            generated_explanation=row["explanation_of_score"],
        )
        response = client.chat.completions.create(
            model=config["model"], temperature=config["temperature"],
            messages=[
                {"role": "system", "content": EXPLANATION_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content
        frame.at[index, "judge_response"] = text
        try:
            parsed = json.loads(text)
            scores = {field: float(parsed[field]) for field in SCORE_FIELDS}
            if not all(0 <= value <= 1 for value in scores.values()):
                raise ValueError("Judge scores must be between zero and one")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            frame.at[index, "judge_status"] = "invalid_response"
            continue
        for field, value in scores.items():
            frame.at[index, field] = value
        frame.at[index, "comment"] = parsed.get("comment", "")
        frame.at[index, "judge_status"] = "complete"
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = load_config(project_path(args.config))
    run_dir = project_path(args.run_dir)
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config)
    frame = pd.read_csv(project_path(config["input_path"]), dtype={"item_id": str})
    evaluated = evaluate_explanations(frame, config)
    evaluated.to_csv(run_dir / "explanation_scores.csv", index=False)
    write_json(run_dir / "results.json", {
        "config": config, "rows": len(evaluated),
        "completed": int(evaluated["judge_status"].eq("complete").sum()),
        "mean_scores": {
            field: float(evaluated[field].mean()) if evaluated[field].notna().any() else None
            for field in SCORE_FIELDS
        },
    })

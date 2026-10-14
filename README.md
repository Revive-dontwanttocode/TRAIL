# Following The TRAIL: Predicting and Explaining Tomorrow’s Hits with a Fine-Tuned LLM

`TRAIL` (*TRend and explanation Integrated Learner*) predicts an item's future popularity
from its description, historical popularity, and a target timestamp. It generates an
explanation and ranks candidates for non-personalized recommendation. Training combines
supervised score prediction with contrastive learning over explanation representations.

![TRAIL architecture](https://files.seeusercontent.com/2026/10/14/9fnF/TRAIL_arc.png)

## Installation

Use Python 3.11 and run the commands from the repository root:

```bash
cd /home/zhixi/codes/trail
conda activate py311recsys
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

If the environment does not exist yet, create it with `conda create -n py311recsys python=3.11`.
The dependency pins target PyTorch 2.8.0 and vLLM 0.11.0. The supplied training configuration
uses LoRA and bfloat16 on CUDA. QLoRA additionally requires `bitsandbytes`, which is not
included in the default dependencies. Model weights and the sentence encoder are loaded
from Hugging Face unless their configured names are replaced with local paths. W&B defaults
to offline mode; change `wandb_config.mode` in the training configuration if needed.

## Configuration and entry points

All paths in JSON configurations are relative to `/home/zhixi/codes/trail`; absolute paths
also work. `extends` is relative to the JSON file containing it and merges nested settings.
Hyperparameters stay in JSON. The Bash launchers accept `CONFIG`, `RUN_NAME`, and, for model
stages, an optional `DEVICE` such as `cuda:0`. Each launcher changes to the repository root
and uses the active Python environment.

| Stage | Bash launcher | Configuration directory |
| --- | --- | --- |
| Normalize existing CSVs | `scripts/prepare_data.sh` | `src/configs/data_preparation.json` |
| Generate training explanations | `scripts/generate_explanations.sh` | `src/configs/explanation_generation/` |
| Fine-tune TRAIL | `scripts/train_trail.sh` | `src/configs/trail_training/` |
| Predict, rank, and optionally evaluate | `scripts/infer_trail.sh` | `src/configs/batch_inference/` |

The model configurations select prompts with `dataset`: `douban`, `baby`, or `beauty`.
All system and user prompts, including the optional explanation judge, live in
[`src/prompts.py`](src/prompts.py). Training, explanation generation, and inference keep
their original stage-specific prompt text. The existing Kitchen prompt is retained;
this repository supplies experiment configurations for Douban, Baby, and Beauty.

## Data preparation

Start from your existing preprocessed CSVs. Keep the original training and inference
periods, popularity values, and histories. The generic preparation command only renames
configured columns and writes the files with their row order and extra columns preserved.
It does not download raw reviews, compute popularity, sample rows, or rebuild time splits.
The repository contains no datasets or example data records. `data/` is ignored by Git.

Edit [`src/configs/data_preparation.json`](src/configs/data_preparation.json): set `dataset`
and the `input_path`, `output_path`, and `columns` for each entry in `files`. Each column
mapping is **canonical name → column name in your source CSV**. Map `item_id` to `movieid`
or `asin` as appropriate, and `item_description` to `movie_description` or
`product_description`. Add mappings for any other renamed fields. Use an empty `columns`
object when a CSV already uses the canonical names.

```bash
CONFIG=src/configs/data_preparation.json RUN_NAME=data_preparation bash scripts/prepare_data.sh
```

The same command handles every dataset. Change its JSON paths and mapping before using it
for another dataset. The default configuration normalizes the two Douban input CSVs into
`data/trail_douban/processed.csv` and `data/trail_douban/inference.csv`.

| Canonical column | Required for | Meaning |
| --- | --- | --- |
| `item_id` | Generation and inference | Original item identifier, read as a string; unique within an inference candidate list |
| `item_description` | All model stages | Movie or product description |
| `historical_popularity_trends` | All model stages | Chronologically ordered JSON list of historical records; each record contains numeric `popularity_score` |
| `predict_timestamp` | All model stages | The target prediction period in the original data |
| `ground_truth_popularity_score` | Training, training-data generation, and optional explanation judging | Original observed popularity label |
| `explanation_of_score` | Training | Generated or previously prepared explanation |

Use one inference candidate list for one target period per run. Ranking evaluation also
needs your held-out user–item interaction CSV for that period; its field names and path
are specified under `evaluation` in the inference configuration. IDs must match the
candidate CSV; no offset or remapping is applied.

If your training CSV already has explanations, normalize it directly to the configured
`training.csv` and skip explanation generation. If your files already match the schema,
set their paths in the model configurations and skip normalization as well.

## Generate training explanations

Set `data.processed_data_path`, `data.inference_result_path`, and `data.training_set_path`
in the dataset's generation configuration. The shared
[`explanation_generation.json`](src/configs/explanation_generation/explanation_generation.json)
contains the model, generation parameters, and backend settings.

```bash
CONFIG=src/configs/explanation_generation/douban_data_generation.json RUN_NAME=douban_explanations bash scripts/generate_explanations.sh
```

Generation reads `processed.csv`, saves raw responses and row status to
`explanation_responses.csv`, and writes accepted explanations to `training.csv`. The
prompt contains only the description, history, and target timestamp. The generated
popularity score is discarded when constructing training data; the original
`ground_truth_popularity_score` remains the supervised label.

Accepted responses need a numeric nonnegative score and the ordered `[Trend]`, `[Feature]`,
and `[Integration]` explanation sections. Failed rows remain in the response CSV and are
excluded from the training CSV. Progress is saved after each batch by default. Rerun the
same command with the same input and output paths to continue pending rows; rows already
marked failed are not retried on resume.

The default backend is `vllm`; it first saves a local backbone with the added explanation
tokens under `vllm.model_path`. This consumes additional disk space. Use a different
checkpoint path if changing the backbone. Set `generation_backend` to `transformers`
to use the existing Transformers generation path. `inference_params.do_sample=false`
means greedy decoding on both backends. Run metadata and progress logs are saved under
`runs/<RUN_NAME>/`, with `generation_progress.json` beside the response CSV.

## Training

```bash
CONFIG=src/configs/trail_training/douban_train.json RUN_NAME=douban_train bash scripts/train_trail.sh
```

The training configuration supplies `dataset_path`, the backbone, sentence encoder,
LoRA settings, contrastive loss settings, and `training_config`. Current defaults use
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `all-mpnet-base-v2`, seed 1025, one epoch, and
LoRA rank 32 / alpha 64. These are the supplied experiment defaults, not a claim of
reproducing the paper's reported results.

The training loader holds out 5% of the supplied training rows using the configured
seed and `huggingface_model_loading_config.split_ratio`. This internal validation split
does not use the inference CSV. Rows whose explanations are truncated past their boundary
tokens are excluded. Training drops an incomplete final batch to keep the batch-local
contrastive calculation and existing visualization valid; validation keeps all its rows.
Supply enough usable training rows for at least one full training batch.

The Trainer evaluates once per epoch by default and records:

- `eval_loss`: teacher-forced cross-entropy on unmasked labels, excluding contrastive loss.
- `eval_token_accuracy`: next-token accuracy over labels other than `-100`.
- `eval_supervised_tokens`: the number of tokens used for that accuracy.

The original masking excludes prompt and explanation tokens. Token accuracy therefore
covers the remaining supervised response tokens, including score and JSON-format tokens;
it is not numerical score accuracy or an explanation-quality measure. The saved adapter
is the final training state; automatic best-epoch selection is not enabled.

The adapter and tokenizer are saved in `runs/<RUN_NAME>/adapter/`. The run also contains
the resolved `config.json`, `results.json` with validation history and environment details,
and `trainer_state.json`. Training and inference require a new `RUN_NAME` for each run.

## Inference and evaluation

Set `testing_dataset_path` and `lora_adapter_path` in the inference configuration. When
using a custom training `RUN_NAME`, update `lora_adapter_path` to its `adapter/` directory.

```bash
CONFIG=src/configs/batch_inference/douban_batch.json RUN_NAME=douban_inference bash scripts/infer_trail.sh
```

Inference uses the Transformers backbone plus the saved LoRA adapter. Its outputs under
`runs/<RUN_NAME>/` are:

| File | Content |
| --- | --- |
| `inference_result.csv` | Original candidate columns, raw response, predicted popularity, and explanation |
| `ranking.json` | Global ranking by predicted popularity, with item ID as the tie-breaker |
| `scores.npz` | Ranked item IDs and their scores |
| `results.json` | Candidate counts, failed-score count, optional ranking metrics, and run metadata |
| `rankings.npz` | Per-user rankings, written when interactions are supplied |
| `last_popularity/` | Corresponding outputs for the training-free last-observed-popularity control |

Responses without a numeric score are retained in the raw CSV and excluded from ranking;
`failed_candidates` records their count. Negative numeric predictions are clipped to zero.
Set `evaluation.user_interaction_path` to enable ranking metrics. Its default is `null`,
which produces predictions and global rankings without user-level metrics.

Evaluation uses the same global list for every user, without filtering previously seen
items. It averages HR, Recall, Precision, standard NDCG, and Reference NDCG over users at
the configured cutoffs (5, 10, 15, and 20 by default). Reference NDCG follows
`recsys_baseline`: each user's standard NDCG is additionally divided by their number of
ground-truth items. Use the `reference_ndcg@K` fields for comparisons with that repository.
The valid, unique candidate ranking must cover the largest requested cutoff. The
`last_popularity` entry in `results.json` records the control under the same metric rules.

## Switch datasets

Use the same launchers with the appropriate configurations and run names:

| Dataset | Generation config in `src/configs/explanation_generation/` | Training config in `src/configs/trail_training/` | Inference config in `src/configs/batch_inference/` | Default data directory |
| --- | --- | --- | --- | --- |
| Douban | `douban_data_generation.json` | `douban_train.json` | `douban_batch.json` | `data/trail_douban/` |
| Amazon Baby | `amazon_baby_data_generation.json` | `amazon_baby_train.json` | `amazon_baby_batch.json` | `data/trail_amazon_baby/` |
| Amazon Beauty | `amazon_beauty_data_generation.json` | `amazon_beauty_train.json` | `amazon_beauty_batch.json` | `data/trail_amazon_beauty/` |

For Baby, after normalizing your existing CSVs to the configured paths:

```bash
CONFIG=src/configs/explanation_generation/amazon_baby_data_generation.json RUN_NAME=baby_explanations bash scripts/generate_explanations.sh
CONFIG=src/configs/trail_training/amazon_baby_train.json RUN_NAME=baby_train bash scripts/train_trail.sh
CONFIG=src/configs/batch_inference/amazon_baby_batch.json RUN_NAME=baby_inference bash scripts/infer_trail.sh
```

For Beauty, use the `amazon_beauty_*` configurations with `beauty_explanations`,
`beauty_train`, and `beauty_inference`. The default inference adapter paths match these
training run names. Baby's configuration overrides the batch size to 4 and accumulation
to 8; Douban and Beauty use 8 and 4. Paths and dataset selection are independent of these
training settings. Record your source dataset release in each model config's
`dataset_version`; the preparation step does not select or convert an Amazon release.

## Optional explanation judging

[`src/run_GPT_judge.py`](src/run_GPT_judge.py) evaluates saved explanations with the
central judge prompt. Edit `input_path`, `model`, and `temperature` in
[`src/configs/explanation_judge.json`](src/configs/explanation_judge.json), and set
`OPENAI_API_KEY` in your environment. This optional command sends each item's description,
history, prediction, explanation, and true popularity score to the configured API model.

```bash
python -m src.run_GPT_judge --config src/configs/explanation_judge.json --run-dir runs/douban_explanation_judge
```

It writes `explanation_scores.csv` and `results.json` with four mean quality scores.
Invalid judge responses retain their raw text and have missing scores instead of zero
scores. This post-hoc judge is separate from generation, training validation, and ranking
evaluation.

## Validation status

This migration has been reviewed by reading the source and diffs. Tests, model execution,
training, evaluation, and dependency installation were not run during the migration.
No new experimental results are claimed. Data, checkpoints, and local run files are
excluded from Git.

## 📄 Citation

```bibtex
@inproceedings{zhang2026trail,
  author = {Zhang, Yinan and Chen, Zhixi and Jing, Jiazheng and Shen, Zhiqi},
  title = {Following the {TRAIL}: Predicting and Explaining Tomorrow's Hits with a Fine-Tuned {LLM}},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  year = {2026},
  pages = {6719--6730},
  doi = {10.1145/3774904.3792616}
}
```

# Following The TRAIL: Predicting and Explaining Tomorrow’s Hits with a Fine-Tuned LLM

[![Paper](https://img.shields.io/badge/Paper-Read-blue)](https://dl.acm.org/doi/10.1145/3774904.3792616)
[![arXiv](https://img.shields.io/badge/arXiv-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.04225)
[![Code](https://img.shields.io/badge/Code-181717?logo=github&logoColor=white)](https://github.com/Revive-dontwanttocode/TRAIL)
[![Citation](https://img.shields.io/badge/Citation-BibTeX-555555)](#-citation)

`TRAIL` (*TRend and explanation Integrated Learner*) predicts an item's future popularity
from its description, historical popularity, and a target timestamp. It generates an
explanation and ranks candidates for non-personalized recommendation. Training combines
supervised score prediction with contrastive learning over explanation representations.

<p align="center">
  TRAIL Paper: <a href="https://dl.acm.org/doi/10.1145/3774904.3792616">https://dl.acm.org/doi/10.1145/3774904.3792616</a>
</p>

![TRAIL architecture](https://files.seeusercontent.com/2026/10/14/9fnF/TRAIL_arc.png)

## ⚙️ Installation

Use Python 3.11 and run the commands from the repository root:

```bash
conda create -n trail python=3.11
conda activate trail
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The dependency pins target PyTorch 2.8.0 and vLLM 0.11.0. The supplied training configuration
uses LoRA and bfloat16 on CUDA. QLoRA additionally requires `bitsandbytes`, which is not
included in the default dependencies. Model weights and the sentence encoder are loaded
from Hugging Face unless their configured names are replaced with local paths.

## 🔧 Configuration and entry points

Paths in JSON configurations are relative to the repository root.

| Stage | Bash launcher | Configuration |
| --- | --- | --- |
| Normalize existing CSVs | `scripts/prepare_data.sh` | `src/configs/data_preparation.json` |
| Generate training explanations | `scripts/generate_explanations.sh` | `src/configs/explanation_generation/` |
| Fine-tune TRAIL | `scripts/train_trail.sh` | `src/configs/trail_training/` |
| Predict, rank, and optionally evaluate | `scripts/infer_trail.sh` | `src/configs/batch_inference/` |

Prompts are defined in [`src/prompts.py`](src/prompts.py).

## 📊 Data preparation and explanation generation

Prepare **Douban**, **Amazon Baby**, or **Amazon Beauty** in two steps: normalize your preprocessed CSVs, then generate training explanations. We trained `TRAIL` on three datasets:

* **Douban Movie**: This is a private dataset that we cannot share till time fruits are available. It contains user-movie interactions and movie metadata.
* **Amazon Baby:** This dataset is publicly available and can be downloaded from [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/).
* **Amazon Beauty:** This dataset is publicly available and can be downloaded from [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/).

Datasets are not bundled with the repository. Supply preprocessed training and inference CSVs with the original popularity values, histories, and time splits; the preparation step standardizes column names without computing popularity or rebuilding splits.

### 1. Normalize the source CSVs

Edit [`src/configs/data_preparation.json`](src/configs/data_preparation.json) for your dataset:

- Set `dataset` to the value in the table above.
- Set `input_path` for `files.training` and `files.inference` to your source CSVs.
- Set their `output_path` to `processed.csv` and `inference.csv`, respectively, in the corresponding data directory.
- Set `columns` using **canonical name → source column name** mappings, such as `item_id` → `movieid` for Douban or `item_id` → `asin` for Amazon. Map `item_description` to `movie_description` or `product_description` as needed; use an empty object if names already match.

```bash
CONFIG=src/configs/data_preparation.json RUN_NAME=data_preparation bash scripts/prepare_data.sh
```

The supplied configuration targets Douban. Update its paths and mappings before preparing
Baby or Beauty. The normalized CSVs use these columns:

| Canonical column | Required for | Meaning |
| --- | --- | --- |
| `item_id` | Generation and inference | Original item identifier, read as a string; unique within an inference candidate list |
| `item_description` | All model stages | Movie or product description |
| `historical_popularity_trends` | All model stages | Chronologically ordered JSON list of historical records; each record contains numeric `popularity_score` |
| `predict_timestamp` | All model stages | The target prediction period in the original data |
| `ground_truth_popularity_score` | Training, training-data generation, and optional explanation judging | Original observed popularity label |
| `explanation_of_score` | Training | Generated or previously prepared explanation |

### 2. Generate explanations and create the training CSV

Run the command for your dataset. Each configuration reads `processed.csv` and writes `explanation_responses.csv` and `training.csv` in the corresponding data directory above.
Update the configuration's `data` paths if you use different locations.

```bash
# Douban
CONFIG=src/configs/explanation_generation/douban_data_generation.json RUN_NAME=douban_explanations bash scripts/generate_explanations.sh

# Amazon Baby
CONFIG=src/configs/explanation_generation/amazon_baby_data_generation.json RUN_NAME=baby_explanations bash scripts/generate_explanations.sh

# Amazon Beauty
CONFIG=src/configs/explanation_generation/amazon_beauty_data_generation.json RUN_NAME=beauty_explanations bash scripts/generate_explanations.sh
```

**Ready for training:** your dataset directory should now contain `training.csv` with usable explanations and `inference.csv` with prediction candidates. The supplied training and inference configurations point to these files.

## 🚀 Training

Run with your dataset's training configuration from the table above and a new `RUN_NAME`:

```bash
CONFIG=src/configs/trail_training/douban_train.json RUN_NAME=douban_train bash scripts/train_trail.sh
```

The trained adapter and tokenizer are saved in `runs/<RUN_NAME>/adapter/`. Run configuration, results, and training state are saved in `runs/<RUN_NAME>/`.

## 🔍 Inference and evaluation

In your dataset's inference configuration, set `testing_dataset_path` to your prepared `inference.csv` and `lora_adapter_path` to the trained `adapter/` directory. To enable ranking evaluation, set `evaluation.user_interaction_path` to the held-out interactions CSV for the same period with matching item IDs.

Run with a new `RUN_NAME`:

```bash
CONFIG=src/configs/batch_inference/douban_batch.json RUN_NAME=douban_inference bash scripts/infer_trail.sh
```

Outputs are saved in `runs/<RUN_NAME>/`: `inference_result.csv` contains predictions and explanations, `ranking.json` contains the item ranking, and `results.json` records run details and evaluation metrics when enabled.

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

## 🙏 Acknowledgments

This research is supported by the RIE2025 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) (Award I2301E0026), administered by A*STAR, as well as supported by Alibaba Group and NTU Singapore through Alibaba-NTU Global e-Sustainability CorpLab (ANGEL). This work is supported by Jinan-NTU Green Technology Research Institute (GreenTRI), and the Joint NTU-UBC Research Centre of Excellence in Active Living for the Elderly (LILY), Nanyang Technological University, Singapore.

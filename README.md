# Following The TRAIL: Predicting and Explaining Tomorrow’s Hits with a Fine-Tuned LLM

This repository contains the `TRAIL` implementation, required python packages, dataset preparation codes,
and bash-style launchers.

## 📖 Overview
`TRAIL` (*TRend and explanation Integrated Learner*) is a fine-tuned large language model for explainable short-term popularity prediction. 

Given an item’s description, historical popularity, and a target timestamp, `TRAIL` predicts its future popularity and generates a natural-language explanation. These predictions are used to rank candidate items for **non-personalized recommendation**.

TRAIL combines supervised fine-tuning with contrastive learning to connect explanation representations with similarities in popularity trends and item characteristics.

![TRAIL_arc.png](https://files.seeusercontent.com/2026/10/14/9fnF/TRAIL_arc.png)

## ⚙️ Installation of Dependencies

Use Linux for the installation of dependencies. For GPU execution, we use `torch` 2.8.0 and `vLLM` 0.11.0 with `CUDA` 12.8 wheels.

Create and activate the environment:

```bash
conda create -n py311recsys python=3.11
conda activate py311recsys
```

Skip environment creation if you already have a suitable environment. Install the dependencies:

```bash
# TODO: Paste the verified dependency installation command here.
```

> TODO: State the working directory for the commands below and any required model access or logging configuration.

## 📊 Dataset Preparation

We trained the model on three datasets:

* **Douban Movie**: This is a private dataset that we cannot share till time fruits are available. It contains user-movie interactions and movie metadata.
* **Amazon Baby:** This dataset is publicly available and can be downloaded from [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/).
* **Amazon Beauty:** This dataset is publicly available and can be downloaded from [Amazon Reviews 2023 Dataset](https://amazon-reviews-2023.github.io/).

You could find the amazon datasets in different categories, such as `Baby`, `Beauty`, `Video Games`, etc. For amazon reviews dataset, we only used the `Baby` and `Beauty` categories in our experiments.

> TODO: Confirm the exact Amazon dataset release and the links to the required review and metadata files.

### Raw Data

> TODO: Show where to place the downloaded files, including their expected filenames.

```text
TODO: Add the raw-data directory layout here.
```

### Preprocessing

```bash
# TODO: Paste the verified preprocessing command here.
```

> TODO: Briefly explain how popularity scores and histories are constructed, how training explanations are obtained, and how training, validation, and test data are split.

Document the files produced by preprocessing:

| File | Path | Required content |
| --- | --- | --- |
| Training data | TODO | TODO: List the required columns and their meanings. |
| Inference data | TODO | TODO: List the required columns and their meanings. |
| Evaluation data | TODO | TODO: Identify the ground-truth labels or interaction files used by evaluation. |

> TODO: Add one small example record so readers can check their prepared data.

## 🚀 Training

> TODO: Specify the base model, training data path, output directory, and the settings used for the reported experiment, including the random seed and contrastive learning configuration.

```bash
# TODO: Paste the verified training command here, including all required arguments.
```

> TODO: Identify the saved model or LoRA adapter and tokenizer files needed for inference, and their output location.

## 🔍 Inference and Evaluation

### Inference

> TODO: Specify the base model, trained adapter or model path, inference data path, generation settings, and prediction output path.

```bash
# TODO: Paste the verified inference command here, including all required arguments.
```

> TODO: Describe the prediction file columns and show one real output containing a popularity score and its explanation.

### Evaluation

> TODO: Specify the prediction file and ground-truth data required for evaluation. Explain which metrics are reported, the values of K, and how results are averaged.

```bash
# TODO: Paste the verified evaluation command here, including all required arguments.
```

> TODO: State where evaluation results are saved and include a short example of the expected output.

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

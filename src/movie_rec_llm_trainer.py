"""
MovieRecLLMTrainer class.
This Trainer class is used to finetune the LLM model. Its fulfill the design of our loss function.

Author: Zhixi Chen
Date: 2025-06-22
"""

from transformers import Trainer

import torch.nn.functional as F
import torch
import gc

import wandb

import numpy as np
import math
import logging

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.evaluation.language_model import causal_lm_token_metrics

from .contrastive_projection_head import ContrastiveProjectionHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def log_matrix_as_heatmap(matrix, name, step=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="Spectral", vmin=0.0, vmax=1.0)
    plt.title(name)
    wandb.log({name: wandb.Image(plt.gcf())}, step=step)
    plt.close()


def plot_embeddings(embeddings, title="t-SNE Embeddings", step=None):
    tsne = TSNE(n_components=2, random_state=88, perplexity=2)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        alpha=0.7
    )
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(False)
    # upload the plot to wandb
    wandb.log({title: wandb.Image(plt.gcf())}, step=step)
    plt.close()


def find_subsequence(main_list, sub_list):
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            return i
    return -1


class MovieRecLLMTrainer(Trainer):
    """
    Custom Trainer for MovieRecLLM.
    This trainer is used to finetune the LLM model for movie recommendation task.
    It will use the `compute_loss` method to calculate the loss for each batch.
    """

    def __init__(self, *args, tokenizer, encoder, contrastive_learning_config, loss_weight_config, trend_similarity_fn,
                 score_similarity_fn, description_similarity_fn, is_contrastive, **kwargs):
        """
        Init the `MovieRecLLMTrainer` with the tokenizer and the contrastive learning configuration.
        In our finetuning process, we will use the `compute_loss` method to calculate the loss for each batch.
        So this trainer will help us to fulfill our design of loss function, and the metrics computation.

        Args:
            *args: Positional arguments for the base Trainer class.
            tokenizer: The tokenizer to be used for tokenizing the input data.
            contrastive_learning_config: Configuration for contrastive learning.
            loss_weight_config: Configuration for loss weights.
            trend_similarity_fn: Function to compute trend similarity.
            score_similarity_fn: Function to compute score similarity.
            description_similarity_fn: Function to compute description similarity.
            **kwargs: Keyword arguments for the base Trainer class.
        """
        # Trainer registers callbacks and the tokenizer as instance attributes.
        kwargs.setdefault("compute_metrics", self.compute_metrics)
        kwargs.setdefault("processing_class", tokenizer)
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.contrastive_learning_config = contrastive_learning_config
        self.loss_weight_config = loss_weight_config
        self.trend_similarity_fn = trend_similarity_fn
        self.score_similarity_fn = score_similarity_fn
        self.description_similarity_fn = description_similarity_fn
        self.is_contrastive = is_contrastive

        # `compute_loss` already averages one batch. Without this, the Trainer of the pinned transformers
        # version does not divide the loss by `gradient_accumulation_steps` for a PEFT model.
        self.model_accepts_loss_kwargs = False
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "hidden_dim", None)

        device = next(self.model.parameters()).device
        if self.is_contrastive:
            self.model.projection_head = ContrastiveProjectionHead(
                in_dim=hidden_size,
                out_dim=self.contrastive_learning_config["projection_out_dim"],
                dropout=self.contrastive_learning_config["projection_dropout"]
            ).to(device)

        # locate popularity score position
        # `self.tokenizer` logs a deprecation warning on every access in the pinned transformers version
        self.prefix_tokens = self.processing_class.encode('"predict_popularity_score": "', add_special_tokens=False)
        self.suffix_tokens = self.processing_class.encode('"', add_special_tokens=False)[0]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ##### 0. Prepare the computational parameters #####
        batch_size = inputs["input_ids"].shape[0]
        top_N = self.contrastive_learning_config["top_N"]
        is_dtw_available = self.contrastive_learning_config["DTW"]
        is_hard_negative_mining = self.contrastive_learning_config["hard_negative_mining"]
        infoNCE_temperature = self.contrastive_learning_config["infoNCE_temperature"]
        hard_negative_mining_top_N = self.contrastive_learning_config["hard_negative_mining_top_N"]
        similarity_weight = self.contrastive_learning_config["similarity_weight"]
        exp_start_token_id = self.processing_class.convert_tokens_to_ids("<exp_start>")
        exp_end_token_id = self.processing_class.convert_tokens_to_ids("<exp_end>")

        if self.state.global_step % 100 == 0:
            logging.info(f"First labels: {inputs['labels'][:, -15:]}")

        ##### 1. Get the model output logits or hidden_states #####
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )

        last_hidden_states = outputs.hidden_states[-1]

        ##### 2. Calculate the score prediction loss by CrossEntropyLoss (just use teacher-forcing) #####
        score_pred_loss = outputs.loss
        logging.info(f"The value of score_pred_loss: {score_pred_loss}")
        logging.info(f"Can the score_pred_loss be back-propagated? {score_pred_loss.requires_grad}")
        logging.info(f"Gradient function: {score_pred_loss.grad_fn}")

        ##### 3. Calculate the contrastive loss #####
        contrastive_loss = 0.0
        if self.is_contrastive:
            with torch.no_grad():
                topN_indices = self.get_topN_similarity_indices(top_N, inputs, is_dtw_available, batch_size,
                                                                similarity_weight)

            embeddings = self.extract_pooled_explanation_embeddings(
                batch_size, last_hidden_states, inputs["input_ids"],
                exp_start_token_id, exp_end_token_id, pooling="mean"
            )
            proj_embeddings = self.model.projection_head(embeddings)
            proj_embeddings = F.normalize(proj_embeddings, p=2, dim=-1)
            emb_sim_matrix = torch.matmul(proj_embeddings, proj_embeddings.T)
            contrastive_loss = self.infoNCE_loss(emb_sim_matrix, topN_indices, infoNCE_temperature, batch_size)
            if self.state.global_step % 100 == 0:
                plot_embeddings(
                    embeddings.detach().cpu().numpy(),
                    title="Normalized Embeddings",
                    step=self.state.global_step
                )
            logging.info(f"The value of contrastive_loss: {contrastive_loss}")
            logging.info(f"Can the contrastive_loss be back-propagated? {contrastive_loss.requires_grad}")
            logging.info(f"Gradient function of contrastive_loss: {contrastive_loss.grad_fn}")

        #### 4. Calculate Record the loss into wandb #####
        final_loss = 0.0
        if self.is_contrastive:
            w1, w2, w3 = self.loss_weight_schedule_v3(
                self.state.global_step, self.state.max_steps,
                warmup_ratio=self.loss_weight_config["contrastive_warmup_ratio"],
                init_w2=self.loss_weight_config["contrastive_init_weight"],
                max_w2=self.loss_weight_config["contrastive_max_weight"]
            )
            final_loss = w1 * score_pred_loss + w2 * contrastive_loss
            wandb.log(
                {
                    "loss/score_pred_loss(CE)": score_pred_loss,
                    "loss/contrastive_loss": contrastive_loss,
                    "loss/final_loss": final_loss,
                    "loss_weight/ce_weight": w1,
                    "loss_weight/contrastive_weight": w2,
                }, step=self.state.global_step
            )
        else:
            # without the contrastive loss, only the score prediction loss is optimized
            final_loss = score_pred_loss
            wandb.log(
                {
                    "loss/score_pred_loss(CE)": score_pred_loss,
                    "loss/final_loss": final_loss,
                }, step=self.state.global_step
            )

        ##### 5. Return the weighted final loss #####
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        if return_outputs:
            return final_loss, {}
        else:
            return final_loss

    def compute_metrics(self, eval_pred):
        """Measure supervised next-token accuracy on the held-out training rows."""
        return causal_lm_token_metrics(eval_pred.predictions, eval_pred.label_ids)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluate supervised CE and return token IDs for the metric callback."""
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        # Validation must also work for a single row, without batch-local positives.
        with torch.no_grad(), self.compute_loss_context_manager():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
        loss = outputs.loss.detach().mean() if outputs.loss is not None else None
        if prediction_loss_only:
            return loss, None, None

        # Reduce before gathering across devices or retaining validation batches.
        prediction_ids = outputs.logits.argmax(dim=-1).detach()
        labels = labels.detach() if labels is not None else None
        return loss, prediction_ids, labels

    def get_eval_dataloader(self, eval_dataset=None):
        """Keep every validation row even when contrastive training drops a tail batch."""
        drop_last = self.args.dataloader_drop_last
        try:
            self.args.dataloader_drop_last = False
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.args.dataloader_drop_last = drop_last

    def safe_float(self, x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def get_topN_similarity_indices(self, top_N, inputs, is_dtw_available, batch_size, similarity_weight) -> list:
        ##### 1. Get description similarity matrix #####
        movie_description_list = list(inputs["movie_description"])
        with torch.no_grad():
            description_similarity_matrix = self.description_similarity_fn(
                movie_description_list,
                self.encoder
            )
        np.fill_diagonal(description_similarity_matrix, -np.inf)

        ##### 2. Get the score similarity matrix #####
        score_similarity_matrix = np.zeros((batch_size, batch_size), dtype=np.float32)
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                score_sim = self.score_similarity_fn(
                    self.safe_float(inputs["ground_truth_popularity_score"][i].cpu()),
                    self.safe_float(inputs["ground_truth_popularity_score"][j].cpu()),
                    inputs["list_historical_popularity_trends"][i],
                    inputs["list_historical_popularity_trends"][j],
                )
                score_similarity_matrix[i, j] = score_sim
                score_similarity_matrix[j, i] = score_sim
        np.fill_diagonal(score_similarity_matrix, -np.inf)

        ##### 3. Calculate the trend similarity matrix #####
        dtw_similarity_matrix = np.zeros((batch_size, batch_size), dtype=np.float32)
        if is_dtw_available:
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    trend_sim = self.trend_similarity_fn(
                        inputs["list_historical_popularity_trends"][i],
                        inputs["list_historical_popularity_trends"][j],
                        normalize=True
                    )
                    dtw_similarity_matrix[i, j] = trend_sim
                    dtw_similarity_matrix[j, i] = trend_sim
        np.fill_diagonal(dtw_similarity_matrix, -np.inf)

        total_similarity_matrix = similarity_weight["description_weight"] * description_similarity_matrix + \
                                  similarity_weight["score_weight"] * score_similarity_matrix + similarity_weight[
                                      "trend_weight"] * dtw_similarity_matrix
        # total_similarity_matrix = similarity_weight["description_weight"] * description_similarity_matrix

        if self.state.global_step % 100 == 0:
            logging.info(f"total similarity matrix: {total_similarity_matrix}")

        if self.state.global_step % 20 == 0:
            log_matrix_as_heatmap(
                total_similarity_matrix,
                "Total Similarity Matrix",
                step=self.state.global_step
            )
            log_matrix_as_heatmap(
                description_similarity_matrix,
                "Description Similarity Matrix",
                step=self.state.global_step
            )
            log_matrix_as_heatmap(
                score_similarity_matrix,
                "Score Similarity Matrix",
                step=self.state.global_step
            )

        ##### 4. Get the top N indices for each sample #####
        topN_indices = []
        for i in range(batch_size):
            _, indices = torch.topk(
                torch.tensor(total_similarity_matrix[i]),
                top_N,
            )
            topN_indices.append(indices)
        return topN_indices

    def extract_pooled_explanation_embeddings(
            self, batch_size: int, hidden_states: torch.Tensor, input_ids: torch.Tensor, exp_start_token_id: int,
            exp_end_token_id: int, pooling: str = "mean"
    ) -> torch.Tensor:
        pooled_embeddings = []
        for i in range(batch_size):
            input_id = input_ids[i]  # get the input_ids of the i-th sample
            hidden = hidden_states[i]  # get the hidden states of the i-th sample

            # find start/end position
            start_pos = (input_id == exp_start_token_id).nonzero(as_tuple=True)[0].item()
            end_pos = (input_id == exp_end_token_id).nonzero(as_tuple=True)[0].item()

            # get the segment
            segment = hidden[start_pos:end_pos + 1, :]

            # do pooling, can add several other methods in the future
            pooled = None
            if pooling == "mean":
                pooled = segment.mean(dim=0)

            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)  # normalize the pooled embedding
            pooled_embeddings.append(pooled)

        return torch.stack(pooled_embeddings, dim=0)

    def get_eos_embedding(self, batch_size: int, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                          eos_token_id: int) -> torch.Tensor:
        eos_embeddings = []
        for i in range(batch_size):
            input_id = input_ids[i]
            hidden = hidden_states[i]
            # find eos positions
            eos_positions = (input_id == eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.numel() > 0:
                eos_pos = eos_positions[-1].item()
            else:
                eos_pos = input_id.size(0) - 1
            eos_emb = hidden[eos_pos, :]  # get the last eos position embedding
            eos_embeddings.append(eos_emb)

        eos_embeddings = torch.stack(eos_embeddings, dim=0)
        return eos_embeddings

    def infoNCE_loss(self, similarity_matrix: torch.Tensor, topN_indices: list, temperature: float, batch_size: int,
                     eps: float = 1e-8):
        device = similarity_matrix.device
        logits = similarity_matrix.to(torch.float32) / temperature  # scale the similarity matrix by temperature
        positive_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
        for i in range(batch_size):
            pos_indices = topN_indices[i].to(device)
            positive_mask[i, pos_indices] = True

        # calculate denominators
        log_denominator = torch.logsumexp(logits, dim=1)

        # calculate numerator
        log_numerator = torch.logsumexp(
            logits.masked_fill(~positive_mask, float('-inf')),
            dim=1
        )

        loss_per_sample = log_denominator - log_numerator
        final_infonce_loss = loss_per_sample.mean()
        return final_infonce_loss

    def loss_weight_scheduler(self, current_step, total_steps, warmup_ratio, init_w1):
        """
        Cosine weights scheduler for the loss calculation.
        """

        min_w1 = 0.4
        warmup_steps = int(total_steps * warmup_ratio)

        if current_step < warmup_steps:
            w1 = init_w1
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(progress * np.pi))
            w1 = min_w1 + (init_w1 - min_w1) * cosine_decay

        w2 = 1.0 - w1

        return w1, w2

    def loss_weight_schedule_v2(self, current_step, total_steps, warmup_ratio=0.1, init_w2=0.2, max_w2=0.8):
        w1 = 1.0

        warmup_steps = int(total_steps * warmup_ratio)
        if current_step < warmup_steps:
            w2 = init_w2 + (max_w2 - init_w2) * (current_step / warmup_steps)
        else:
            anneal_progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            w2 = init_w2 + (max_w2 - init_w2) * 0.5 * (1 + math.cos(math.pi * anneal_progress))
        return w1, w2

    def loss_weight_schedule_v3(self, current_step, total_steps, warmup_ratio=0.1, init_w2=0.2, max_w2=0.8):
        w1 = 1.0
        w3 = 0.4

        warmup_steps = int(total_steps * warmup_ratio)
        if current_step < warmup_steps:
            w2 = init_w2 + (max_w2 - init_w2) * (current_step / warmup_steps)
        else:
            anneal_progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            w2 = init_w2 + (max_w2 - init_w2) * 0.5 * (1 + math.cos(math.pi * anneal_progress))
        return w1, w2, w3
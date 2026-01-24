import torch
from torch.nn.utils.rnn import pad_sequence


class DataCollatorForMovieRec:
    """
    Compact the list of data into a batch, and add padding to `input_ids`, `attention_mask`, and `labels`.

    As for other information, such as `movieid`, `movie_description`, and `popularity_history`, they are not padded, and saved as they were.
    So that we can use them for calculating the loss.

    Args:
        tokenizer (`PreTrainedTokenizer`): The tokenizer to use for padding.
        padding (`str` or `bool`, *optional*): Padding strategy. Default is "longest".
        max_length (`int` or `None`, *optional*): Maximum length of the sequences. Default is None.
        return_tensors (`str`, *optional*): The type of tensors to return. Default is "pt".
    """

    def __init__(
            self,
            tokenizer,
            padding: str = "longest",
            max_length = None,
            return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors
        # set padding token at right side
        self.tokenizer.padding_side = "left"

    def left_padding(self, tensor, max_length, pad_value):
        pad_len = max_length - tensor.size(0)
        if pad_len <= 0:
            return tensor
        padding = torch.cat(
            [torch.full((pad_len,), pad_value, dtype=tensor.dtype), tensor]
        )
        return padding

    def __call__(self, features: list) -> dict:
        """
        This function defined how to obtain a list of data samples and transform them into one batch.
        And, this batch data can be directly used for training.

        Args:
            features (`list`): A list of data samples.

        Returns:
            batch (`dict`): A dictionary containing the batch data with suitable padding.
        """

        # Define a safe float conversion function to handle potential conversion errors
        def safe_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        lengths = [len(feature["input_ids"]) for feature in features]
        max_length = max(lengths) if self.max_length is None else min(max(lengths), self.max_length)

        # left padding
        input_ids = [self.left_padding(
            torch.tensor(feature["input_ids"], dtype=torch.long), max_length, self.tokenizer.pad_token_id)
            for feature in features]
        attention_mask = [self.left_padding(
            torch.tensor(feature["attention_mask"], dtype=torch.long), max_length, 0)
            for feature in features]
        labels = [self.left_padding(
            torch.tensor(feature["labels"], dtype=torch.long), max_length, -100)
            for feature in features]

        batch = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

        batch["ground_truth_popularity_score"] = torch.tensor(
            [safe_float(feature["ground_truth_popularity_score"]) for feature in features],
            dtype=torch.float32
        )
        batch["historical_popularity_trends"] = [feature["historical_popularity_trends"] for feature in features]
        batch["predict_timestamp"] = [feature["predict_timestamp"] for feature in features]
        batch["movie_description"] = [feature["movie_description"] for feature in features]
        batch["list_historical_popularity_trends"] = [
            feature["list_historical_popularity_trends"] for feature in features
        ]
        return batch


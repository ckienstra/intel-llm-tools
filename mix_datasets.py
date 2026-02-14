# mix_datasets.py
#
# The goal of this file is to consume a json config of dataset names
# and mix them by percentage for model calibration.
#
import argparse
import json
import logging
import math
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# The json will look like:
# {
#   "samples": 512,
#   "seqlen": 2048,
#   "datasets": [
#     {
#       "name": "PygmalionAI/PIPPA",
#       "percentage": 0.5,
#       "type": "json",
#       "data_files": "https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/pippa_deduped.jsonl",
#       "text_field": "conversation"
#     },
#     {
#       "name": "NeelNanda/pile-10k",
#       "percentage": 0.5,
#       "split": "train",
#       "text_field": "text"
#     }
#   ]
# }

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Mix datasets from a JSON config."
)
parser.add_argument("config_file", help="Path to the JSON config file.")
args: argparse.Namespace = parser.parse_args()

config: dict = json.load(open(args.config_file, "r"))


class Error(Exception):
    """
    Base class for exceptions
    """

    pass


class InvalidConfigurationError(Error):
    """
    Raised when the configuration is invalid
    """

    pass


class DatasetMixer:
    """
    A class to handle the loading and mixing of multiple datasets
    based on a provided configuration.
    """

    def __init__(
        self,
        logger,
        tokenizer,
        config,
    ) -> None:
        self._logger: logging.Logger = logger
        self._tokenizer: AutoTokenizer = tokenizer
        self._config: dict = config
        self.calibration_data: list[str] = []

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def config(self) -> dict:
        return self._config

    def validate(self) -> None:
        percentage_sum = 0
        total_samples = self.config.get("samples", 0)
        if total_samples < 0 or total_samples > 1024:
            raise InvalidConfigurationError(
                "Total samples must be between 0 and 1024. Got: %s",
                total_samples,
            )

        for dataset in self.config.get("datasets", []):
            name = dataset.get("name")
            if not name or "/" not in name:
                raise InvalidConfigurationError(
                    "All datasets must have a HuggingFace dataset name. "
                    "Got: %s",
                    name,
                )
            percentage = dataset.get("percentage")
            if percentage is None:
                raise InvalidConfigurationError(
                    f"Dataset '{name}' is missing 'percentage' key."
                )

            # percentage == 0 is allowed if you want to easily exclude a
            # dataset during quick iteration of your config without
            # removing its entire config stanza.
            if not (0 <= percentage <= 1):
                raise InvalidConfigurationError(
                    "All datasets must have a percentage between 0 and 1. "
                    "Got: %s",
                    percentage,
                )
            percentage_sum += percentage

        # Use math.isclose for floating point comparison
        if not math.isclose(percentage_sum, 1.0):
            raise InvalidConfigurationError(
                "Dataset percentages must sum to 1. Got: %s", percentage_sum
            )

    def mix(self) -> list[str]:
        """
        Processes the datasets defined in the config and returns a mixed list of samples.
        """
        total_samples = self.config.get("samples", 512)
        min_seqlen = self.config.get("seqlen", 2048)
        mixed_data = []

        for ds_info in self.config.get("datasets", []):
            name: str = ds_info.get("name")
            percentage: float = ds_info.get("percentage", 0.0)
            if percentage == 0:
                self.logger.info(f"Skipping {name} as its percentage is 0.")
                continue

            num_samples_for_ds = int(total_samples * percentage)
            if num_samples_for_ds == 0:
                continue

            # FIXME: Would it be possible to fetch more samples while
            # guaranteeing that all samples are unique?
            # I'm multiplying by 20. It usually requires MANY samples to
            # fetch the required number at the minimum sequence length due to
            # the number of very short samples.
            max_items_to_check = num_samples_for_ds * 20

            self.logger.info(
                f"Attempting to find {num_samples_for_ds} samples from {name}..."
            )

            load_args = {"path": name, "split": ds_info.get("split", "train")}
            if "data_files" in ds_info:
                load_args["path"] = ds_info.get("type", "json")
                load_args["data_files"] = ds_info["data_files"]

            try:
                # streaming=True is more memory efficient for large datasets
                dataset = load_dataset(**load_args, streaming=True)
            except Exception as e:
                self.logger.error(f"Failed to load dataset {name}. Error: {e}")
                continue

            text_field = ds_info.get("text_field")
            if not text_field:
                self.logger.error(
                    f"Config for dataset {name} is missing 'text_field'. Skipping."
                )
                continue

            progress_bar = tqdm(
                total=num_samples_for_ds, desc=f"Processing {name}"
            )
            found_samples = 0
            for item in dataset.take(max_items_to_check):
                if found_samples >= num_samples_for_ds:
                    break

                text = item.get(text_field, "")
                # Handle PIPPA's conversational format
                if isinstance(text, list):
                    convo_text = ""
                    for turn in text:
                        role = "User" if turn.get("is_human") else "Bot"
                        message = turn.get("message", "").strip()
                        if message:
                            convo_text += f"{role}: {message}\n"
                    text = convo_text

                if text and isinstance(text, str):
                    # AutoTokenizer is untyped, unfortunately, but it can provide a callable
                    # if it is instantiated and "from_pretrained" is called on it.
                    token_count = len(
                        self.tokenizer(text, add_special_tokens=True)[
                            "input_ids"
                        ]
                    )  # pyright: ignore[reportCallIssue]
                    if token_count >= min_seqlen:
                        mixed_data.append(text)
                        progress_bar.update(1)
                        found_samples += 1
            progress_bar.close()

        return mixed_data

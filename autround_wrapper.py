"""A wrapper for Intel AutoRound for my personal home PC."""

import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any

# auto_round is in this environment, but VSCode doesn't like it for some reason
from auto_round import AutoRound  # type: ignore
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# Example output:
# .\quantize-heretic-quality.py \
#   --model-name coder3101/Cydonia-24B-v4.3-heretic-v2 \
#   --formats fake \
#   --dataset pippa \
#   --output-dir .\Cydonia-Pippa-FP16
#   --interactive;

# 1. Configuration
torch.set_float32_matmul_precision("high")
# I don't know how to get the local path working properly yet
# model_name = "C:\Users\Chris\.cache\huggingface\hub\models--coder3101--Cydonia-24B-v4.3-heretic-v2\snapshots\4e48125d201aaeeb6d6f9349c8c75e772fcfbb25"  # noqa: E501 # pylint: disable=line-too-long
default_model_name = "coder3101/Cydonia-24B-v4.3-heretic-v2"
default_output_dir = Path(".") / (
    default_model_name.rsplit("/", maxsplit=1)[-1] + "-quantized-auto-round"
)


def setup_logger(name: str = "auto_round_quantization") -> logging.Logger:
    """Configure and return a logger. Safe to call multiple times."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler at INFO level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    fmt = "%(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)

    # Avoid adding duplicate handlers when module is reloaded/imported
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(ch)
    logger.propagate = False

    return logger


def save_formats(
    # TODO: Type 'AutoRound' seems problematic.
    autoround: Any,
    logger: logging.Logger,
    formats: list[str],
    output_dir: Path,
    interactive: bool = False,
) -> None:
    """Save the quantized model in requested formats."""
    final_formats: dict[str, bool] = {fmt: False for fmt in formats}
    if not final_formats:
        logger.info("No formats specified; nothing to save.")
        return
    while True:
        try:
            for fmt in list(final_formats.keys()):
                if final_formats[fmt]:
                    continue
                output_path = Path(output_dir) / fmt
                autoround.save_quantized(output_dir=output_path, format=fmt)
                print("\nCopying original model metadata.")
                extensions = [
                    "*.json",
                    "*.jinja",
                    "*.txt",
                    "*.model",
                    "*.tiktoken",
                ]
                for ext in extensions:
                    # TODO: This attribute is private.
                    # # Find a better way to get the original model path.
                    original_model_path = Path(
                        # TODO: Find a public means of getting this path.
                        autoround.model.config._name_or_path  # noqa: E501 # pylint: disable=protected-access,line-too-long
                    )
                    for src in original_model_path.glob(ext):
                        shutil.copy(src, output_path)
                # AutoRound only copies some files by default.
                # Copy all of them for tools further down the pipeline.
                final_formats[fmt] = True
            break
        except Exception as e:  # pylint: disable=broad-except
            # Saving can fail for many reasons; log and allow the caller to
            # decide whether to retry or exit. In non-interactive mode we
            # do not prompt, to avoid blocking automated runs.
            logger.error(f"Failed to save quantized model. Exception: {e}")
            for fmt, completed in final_formats.items():
                status: str = "FAILED" if not completed else "SUCCESS"
                logger.error(f"  {fmt} - {status}")
            if not interactive:
                logger.info("Non-interactive mode: aborting format save loop.")
                break
            add_formats: str = input(
                "If you'd like to try saving to any other formats before exit,"
                " enter them here. "
                "i.e. auto_gptq,auto_awq or press Enter to exit: "
            )
            if not add_formats.strip():
                break
            for fmt in [
                f.strip() for f in add_formats.split(",") if f.strip()
            ]:
                final_formats.setdefault(fmt, False)
    logger.info("Format summary:")
    for fmt, completed in final_formats.items():
        status: str = "FAILED" if not completed else "SUCCESS"
        logger.info(f"  {fmt} - {status}")


def get_pippa_calibration_data(
    # NOTE: It should be an AutoTokenizer, but "from_pretrained" turns
    # it into an Unknown.
    tokenizer: Any,
    num_samples: int = 512,
    min_seqlen: int = 2048,
) -> list[str]:
    """Fetches and formats the PygmalionAI/PIPPA dataset for calibration.

    This function downloads the PIPPA dataset, formats conversations into
    a single string per sample, and returns a list of strings. It filters
    out samples that are shorter than `min_seqlen` tokens to ensure the
    calibration set is valid for the quantization library.
    """
    logger = logging.getLogger("auto_round_quantization")
    logger.info(
        "Fetching and processing %s samples from "
        "PygmalionAI/PIPPA with min length %s tokens.",
        num_samples,
        min_seqlen,
    )

    # We need to iterate through the dataset until we find enough long samples.
    # Set a max number of items to check to avoid an infinite loop.
    # FIXME: A 5% success rate for long samples is a reasonable guess for
    # PIPPA, but this is just a guess :-)
    max_items_to_check = num_samples * 20
    try:
        # Use streaming to avoid downloading the entire massive dataset
        dataset = load_dataset(
            "json",
            data_files="https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/pippa_deduped.jsonl",  # noqa: E501 # pylint: disable=line-too-long
            split="train",
            streaming=False,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(
            "Failed to load PIPPA dataset. Make sure 'datasets' and 'tqdm' "
            "are installed (`pip install datasets tqdm`). Error: %s",
            e,
        )
        return []

    calibration_data = []
    # .take() is efficient with streaming datasets
    dataset_subset = dataset.take(max_items_to_check)

    pbar = tqdm(total=num_samples, desc="Finding long PIPPA samples")
    for data_item in dataset_subset:
        if len(calibration_data) >= num_samples:
            break  # We have enough

        conversation_text = ""
        # If data_item is a Mapping, MutableMapping, or dict, then
        # __getitem__ should exist. If not, isinstance won't be satisfied.
        # Then we can check if the value is a list type.
        if "conversation" not in data_item or not isinstance(  # type: ignore
            data_item["conversation"], list  # type: ignore
        ):
            continue

        for turn in data_item["conversation"]:  # type: ignore
            if (
                not isinstance(turn, dict)
                or "is_human" not in turn
                or "message" not in turn
            ):
                continue

            role = "User" if turn["is_human"] else "Bot"
            message = turn.get("message")

            if message and isinstance(message, str) and message.strip():
                conversation_text += f"{role}: {message.strip()}\n"

        if conversation_text:
            # Check token count before adding to the list.
            # Use truncation=False to get the true length.
            token_count = len(
                tokenizer(conversation_text, truncation=False)["input_ids"]
            )
            if token_count >= min_seqlen:
                calibration_data.append(conversation_text)
                pbar.update(1)
    pbar.close()

    if len(calibration_data) < num_samples:
        logger.warning(
            "Only able to find %s / %s "
            "samples long enough (>= %s tokens) after checking "
            "%s records.",
            len(calibration_data),
            num_samples,
            min_seqlen,
            max_items_to_check,
        )

    logger.info(
        "Successfully processed %s samples from PIPPA.", len(calibration_data)
    )
    return calibration_data


def main() -> None:
    """Do the actual work of running AutoRound."""
    logger = setup_logger()

    # Parse CLI early so flags can influence behavior before heavy work begins.
    parser = argparse.ArgumentParser(
        description="Quantize a model using AutoRound and save in multiple "
        "formats."
    )
    parser.add_argument(
        "--model-name",
        default=default_model_name,
        help="Model identifier, e.g. owner/model "
        "(must contain a forward slash).",
    )
    parser.add_argument(
        "--formats",
        default="auto_gptq,auto_awq,auto_round,fake",
        help="Comma-separated list of formats to save",
    )
    parser.add_argument(
        "--dataset",
        default="pile",
        choices=["pile", "pippa", "c4"],
        help='Calibration dataset to use. "pile" for NeelNanda/pile-10k, '
        '"pippa" for PygmalionAI/PIPPA, "c4" for allenai/c4.',
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=512,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Where to store quantized outputs"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Allow interactive prompts on errors",
    )
    args = parser.parse_args()

    # Validate and use the provided model name
    model_name = args.model_name
    if "/" not in model_name:
        logger.error(
            "Invalid --model-name: must contain a forward slash (owner/model)."
        )
        sys.exit(2)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(".") / (model_name.split("/")[-1] + "-quantized-auto-round")
    )

    logger.info("Loading tokenizer for %s.", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # AutoRound's default sequence length for calibration is 2048. We must
    # pre-filter the calibration dataset to provide samples that meet this
    # length requirement, otherwise AutoRound will discard them and fail.
    autoround_seqlen = 2048

    # Select and prepare the calibration dataset
    if args.dataset == "pippa":
        calibration_dataset = get_pippa_calibration_data(
            tokenizer=tokenizer,
            num_samples=args.nsamples,
            min_seqlen=autoround_seqlen,
        )
        if not calibration_dataset:
            logger.error("Failed to get PIPPA calibration data. Exiting.")
            sys.exit(1)
    else:
        # The default dataset used by AutoRound
        calibration_dataset = "NeelNanda/pile-10k"

    # A VC Build Tools update can update the version number in this path
    os.environ["PATH"] += (
        os.pathsep
        + r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"  # noqa: E501 # pylint: disable=line-too-long
    )
    # 2. Load the model directly to the RTX 5090
    # 24B parameters in BF16 takes ~48GB.
    # Saving to "fake" format requires the 2x of full F16 model size in RAM,
    # so 96GB for 24B model. It might be smart to run a memory checker
    # before starting quantization in case there isn't enough RAM + Pagefile.
    # Maybe:
    # - Warn if lack of RAM the will cause swapping
    # - Raise error if not enough total memory (RAM + Pagefile) to succeed.
    # That prevents hours of wasted time only to fail at the end.
    # Also make an option to save output to a log.
    #
    # Additionally, some files in the original folder need to be copied to the
    # quantized model folder for it to work properly. AutoRound does some, but
    # not all of this automatically. Add a flag --copy-original-file=<filename>
    # which can be specified multiple times to copy specific files from the
    # original model folder to the quantized model folder.
    # Cydonia, a Mistral-based model, needs special_tokens_map.json copied
    # to the output folder.
    # Additionally x3, AutoRound messes up config.json, so also copy that after
    # quantization is complete.

    logger.info("Loading %s to System RAM.", model_name)
    # Use device_map={'': 'cpu'} to ensure no weights are left on the 'meta'
    # device.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cpu"},
    )

    # 3. Initialize AutoRound for CUDA
    # enable_torch_compile=True works well on RTX 50-series/40-series cards.
    # I need to launch this script from within a Visual Studio 2022 environment
    # for all the optimizations to work. This is not an optimization that would
    # work on the server, so I am not going to waste any more time on it.
    #
    # TODO: Add a --low-memory flag which introduces:
    #  gradient_accumulate_steps=8
    #  batch_size=1
    # unfortunately, it slows quantization time by ~50%-100%,
    # but it can save up to 25% of system memory.
    # https://github.com/intel/auto-round/blob/main/docs/step_by_step.md#quantization-costs
    # Additionally, how ridiculous would it be to check the file size of the
    # model, look at VRAM size, and decide whether to enable low_gpu_mem_usage
    # automatically?
    logger.info("Initializing AutoRound quantization...")
    autoround = AutoRound(
        model,
        tokenizer,
        dataset=calibration_dataset,
        scheme="W4A16",
        nsamples=args.nsamples,
        sym=True,
        iters=1000,
        low_gpu_mem_usage=True,
        enable_torch_compile=True,  # Requires VSCode 2022 environment
        device_map="cuda",
    )

    logger.info("Quantizing and saving quantized model to %s.", out_dir)
    logger.info(
        "This may take several hours depending on the model size and hardware."
    )
    logger.info("You will receive this error: ")
    logger.info(
        "this API is deprecated; this script will continue to use "
        "AutoRound.quantize() as it is compatible with current environment"
    )
    logger.info("This is expected and should be ignored.")
    # Ignore the deprecation warning.
    autoround.quantize()

    save_formats(
        autoround,
        logger,
        formats,
        output_dir=out_dir,
        interactive=args.interactive,
    )
    logger.info("Auto-round quantization process complete.")
    logger.info("Quantized model saved to: %s.", out_dir)


if __name__ == "__main__":
    main()

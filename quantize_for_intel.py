"""quantize_for_intel.py.

This script performs a multi-stage conversion and quantization process to
prepare a HuggingFace FP16 model for efficient inference on Intel hardware
using OpenVINO.
The process includes:
1. Converting the HuggingFace FP16 model to OpenVINO-IR format using the
optimum-cli export tool.
2. Sorting the model's layers based on a user-defined mapping of layer name
substrings to quantization precision levels (FP16, INT8_SYM, INT4_SYM).
3. Applying NNCF weight quantization to the model according to the sorted
layers, while ignoring layers that should remain in higher precision.
4. Converting and saving the tokenizer in OpenVINO format, with special
handling for Mistral models that require a regex fix.
5. Saving the final quantized model and tokenizer to disk, and performing a
sanity check on the final model size to verify that quantization was applied
as expected.
The script is designed to be configurable through parameters defined at the
top, and includes error handling and logging for better traceability of the
conversion process.
"""

import logging
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import NamedTuple

import nncf
import openvino as ov
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("nncf").setLevel(logging.INFO)

# --- CONFIGURATION --- #
# TODO: Turn all of this into API parameters.
# TODO: I should have a CLI file to convert args to API parameters.
fake_fp16_dir: Path = Path(
    "/media/raid/Personal_Directories/Chris/LLMs/Models/"
    "Cydonia-24B-Unquantized/snapshots/"
    "4e48125d201aaeeb6d6f9349c8c75e772fcfbb25"
)
# TODO: This is a temp dir. Make a per-run random seed
# subdir in this location. Then, deletion is scoped.
intermediate_dir: Path = Path(
    "/media/Sandisk-500G/kubernetes-volumes/llm-quantization/Cydonia-Temp-Dir"
)
final_dir: Path = Path(
    "/media/Sandisk-500G/kubernetes-volumes/llm-quantization/"
    "Cydonia-Mixed-Quant-RtN"
)
LAYER_PRECISION_MAPPING: dict[str, list[str]] = {
    "INT8_SYM": [
        "embed",  # FP16 causes precision desync
        "lm_head",  # Keeps the final word choice stable
        "down_proj",  # The "Instruction" layer (prevents the User: tags)
    ],
    "INT4_SYM": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
    ],
}

# TODO: Make sure this works as FP16.
UNKNOWN_QUANTIZATION_LEVEL: str = "INT8_SYM"
QLEVEL_STR_TO_NNCF_MODE: dict[str, nncf.CompressWeightsMode] = {
    "INT8_SYM": nncf.CompressWeightsMode.INT8_SYM,
    "INT4_SYM": nncf.CompressWeightsMode.INT4_SYM,
}


class LayerSortingResult(NamedTuple):
    """A structured way to return the results of layer sorting."""

    param_counts: dict[str, int]
    layer_map: dict[str, list[str]]


class Error(Exception):
    """Base exception class for this module."""


class CLIExportError(Error):
    """Raised when the optimum-cli export command fails."""


def model_size_bytes(
    input_dir: Path,
) -> int:
    """Estimate the memory footprint of the input model.

    This is a rough, hacky estimate! It should be replaced with something
    more reliable.

    Args:
        input_dir (Path): The location on disk of the OpenVINO-IR model
                          to be quantized.
    """
    cmd: str = f"du -sh --bytes {input_dir} | cut -f 1"
    output: subprocess.CompletedProcess = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        check=False,
        text=True,
    )
    if output.returncode != 0:
        logging.error(
            "HuggingFace Accelerate command failed. If it is not installed, "
            "run: `pip install accelerate`.",
        )
        return -1
    size: int = -1
    try:
        size = int(output.stdout.strip())
    except (ValueError, TypeError) as e:
        logging.error(
            f"Unable to parse output of command: `{cmd}`\n"
            f"Output: {output.stdout.strip()}\nError: {e}"
        )
        return -1
    return size


def sort_layers_by_precision(
    ov_model: ov.Model,
    layer_precision_map: dict[str, list[str]],
    unknown_quantization_level: str,
) -> LayerSortingResult:
    """Sort model layers by quantization precision level.

    Analyzes the OpenVINO model to categorize layers based on the provided
    precision mapping, assigning each layer a quantization level (FP16,
    INT8_SYM, INT4_SYM) based on substring matching of layer names.

    Args:
        ov_model (ov.Model): The OpenVINO model to analyze.
        layer_precision_map (dict[str, list[str]]): Mapping of quantization
            levels to lists of layer name substrings for categorization.
        unknown_quantization_level (str): Default quantization level to
            assign to layers that don't match any precision mapping.

    Returns:
        LayerSortingResult: Named tuple containing parameter counts per
            quantization level and mapping of layers to their assigned levels.
    """
    logging.info("\nSorting model layers by quantization precision.")
    param_counts: dict[str, int] = {
        "FP16": 0,
        "INT8_SYM": 0,
        "INT4_SYM": 0,
    }
    layer_map: dict[str, list[str]] = {
        "FP16": [],
        "INT8_SYM": [],
        "INT4_SYM": [],
    }
    for op in ov_model.get_ops():
        if op.get_type_name() != "Constant":
            logging.debug(
                'Skipping non-constant op "%s" of type "%s".',
                op.get_friendly_name(),
                op.get_type_name(),
            )
            continue

        # The 'layer' for NNCF is the operation that consumes the constant
        # weight, not the constant itself. We find the consumer node to get
        # the correct name for the ignored_scope.
        output_port = op.output(0)
        target_inputs = output_port.get_target_inputs()
        if not target_inputs:
            logging.debug(
                "Constant op '%s' has no consumers. Skipping.",
                op.get_friendly_name(),
            )
            continue

        # This iterator will be useful later when listing inputs.
        consumer_op: ov.Node = next(iter(target_inputs)).get_node()
        # NNCF's ignored_scope expects the full node name. Whatever the final
        # node is, we'll use its "friendly name" to populate ignored_scope.
        layer_name: str = consumer_op.get_friendly_name()

        # If the immediate consumer is a Convert or Gather operation, it is
        # likely an intermediate step. The actual layer to target is the
        # consumer of this operation.
        consumer_type: str = consumer_op.get_type_name()
        if consumer_type in ["Convert", "Gather"]:
            logging.debug(
                "Found intermediate op '%s' of type %s. "
                "Looking for its consumer.",
                consumer_op.get_friendly_name(),
                consumer_type,
            )
            # Using list() to exhaust the iterator and check its contents
            real_consumer_inputs: list[ov.Output] = list(
                consumer_op.output(0).get_target_inputs()
            )
            if real_consumer_inputs:
                # The real consumer is the node we want to name.
                layer_name = (
                    real_consumer_inputs[0].get_node().get_friendly_name()
                )
            else:
                # Falls back to original layer_name definition.
                logging.debug(
                    "Intermediate op '%s' has no consumers. "
                    "Using op's name as a fallback.",
                    consumer_op.get_friendly_name(),
                )

        # Get parameter count (size) of the weight constant
        params: int = math.prod(op.get_output_tensor(0).get_shape())

        # This provides substring matching from user-defined layer
        # names to model layer names.
        # e.g. "q_proj" in "__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul"  # noqa: E501 # pylint: disable=line-too-long
        # We use lower() for case-insensitive matching against the map, but
        # store the original name.
        # FIXME: Multi-line tuples are gross, but so is the unwrapped logic :-(
        if any(
            x in layer_name.lower() for x in layer_precision_map.get("FP16", [])
        ):
            layer_map["FP16"].append(layer_name)
            param_counts["FP16"] += params
        elif any(
            x in layer_name.lower()
            for x in layer_precision_map.get("INT8_SYM", [])
        ):
            layer_map["INT8_SYM"].append(layer_name)
            param_counts["INT8_SYM"] += params
        elif any(
            x in layer_name.lower()
            for x in layer_precision_map.get("INT4_SYM", [])
        ):
            layer_map["INT4_SYM"].append(layer_name)
            param_counts["INT4_SYM"] += params
        else:
            logging.debug(
                "Unknown quantization level for op: "
                '"%s" with %s parameters. '
                "Assigning to %s.",
                op.get_friendly_name(),
                params,
                unknown_quantization_level,
            )
            layer_map[unknown_quantization_level].append(layer_name)
            param_counts[unknown_quantization_level] += params

    # This output is really just for fun.
    print("Parameter Count by Quantization Precision:")
    total_params = sum(param_counts.values())
    for qlevel, qlevel_params in param_counts.items():
        percentage = (
            (qlevel_params / total_params) * 100 if total_params > 0 else 0
        )
        print(f"  {qlevel}: {qlevel_params} parameters ({percentage:.2f}%)")
    return LayerSortingResult(param_counts=param_counts, layer_map=layer_map)


def format_and_save_tokenizer(
    input_dir: Path,
    output_dir: Path,
) -> None:
    """Convert and save the tokenizer in OpenVINO format.

    Loads the tokenizer from the input directory and converts it to OpenVINO-IR
    format. Applies special handling for Mistral models that require a regex
    fix. Saves both tokenizer and detokenizer models if available.

    Args:
        input_dir (Path): The location on disk of the HuggingFace model
                          containing the tokenizer to convert.
        output_dir (Path): The location on disk where the converted tokenizer
                           should be saved.

    Raises:
        FileNotFoundError: If the config.json file is not found in input_dir.
    """
    # It is safe to use fix_mistral_regex on non-mistral models.
    # HuggingFace Transformers only applies the fix to models it recognizes as
    # "older" mistral-base model names.
    tokenizer = AutoTokenizer.from_pretrained(
        input_dir, trust_remote_code=True, fix_mistral_regex=True
    )
    ov_tokenizer: ov.Model | tuple[ov.Model, ov.Model] = convert_tokenizer(
        tokenizer, with_detokenizer=True
    )
    if isinstance(ov_tokenizer, tuple):
        ov.save_model(ov_tokenizer[0], output_dir / "openvino_tokenizer.xml")
        ov.save_model(ov_tokenizer[1], output_dir / "openvino_detokenizer.xml")
    else:
        ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")


def convert_hf_to_openvino_ir(input_dir: Path, output_dir: Path) -> None:
    """Convert the HuggingFace FP16 model into an OpenVINO-IR FP16 model.

    This is to make it
    compatible with the Optimum-Intel API. This conversion must be done before
    quantization.
    Save the model to disk to avoid keeping 2x the memory footprint of
    the FP16 model in system memory.

    Args:
        input_dir (Path): The location on disk of the FP16 model to be
                          converted.
        output_dir (Path): The location on disk where the converted model
                           should be saved.
    """
    logging.info("\nConverting HuggingFace model to OpenVINO.")
    # Can we get a loading bar here?
    # It just hangs for a while during export with no user feedback.
    # Perhaps we could trick tqdm by comparing input vs output sizes?
    # The sizes should be identical, since they should both be FP16.
    # Additionally, we should ensure the disk has enough space to save an
    # extra FP16 copy of the model for export.
    cmd = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        str(input_dir),
        "--task",
        "text-generation-with-past",
        "--weight-format",
        "fp16",  # Export as standard FP16
        "--trust-remote-code",
        str(output_dir),
    ]
    logging.info("Executing: %s", " ".join(cmd))
    input_size = model_size_bytes(input_dir)
    if input_size <= 0:
        logging.warning(
            "Could not determine input model size. "
            "Skipping progress bar for conversion."
        )
        try:
            # Run without progress bar, but capture output for better errors.
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logging.error("optimum-cli export stdout:\n%s", e.stdout)
            logging.error("optimum-cli export stderr:\n%s", e.stderr)
            raise CLIExportError(
                f"CRITICAL: optimum-cli Export failed: {e}",
            ) from e
        return

    # We redirect stdout to DEVNULL to prevent optimum-cli's own progress
    # indicators from interfering with our tqdm bar. Stderr is captured to
    # be displayed in case of an error.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )

    def progress_monitor(progress_bar: tqdm) -> None:
        """Monitor the output directory size and update the progress bar."""
        last_size = 0
        while process.poll() is None:
            if not output_dir.exists():
                time.sleep(1)
                continue
            current_size = model_size_bytes(output_dir)
            update_amount = current_size - last_size
            if update_amount > 0:
                progress_bar.update(update_amount)
                last_size = current_size
            # Poll every second.
            # Sometimes `du` will hang for a few seconds, so
            # regular updates help show progress.
            time.sleep(1)

        # One final update to catch writes that happened after the last poll
        if output_dir.exists():
            current_size = model_size_bytes(output_dir)
            update_amount = current_size - last_size
            if update_amount > 0:
                progress_bar.update(update_amount)

    # Spawn the progress_monitor thread in a context, just in case.
    with tqdm(
        total=input_size,
        mininterval=1,
        unit="B",
        unit_scale=True,
        desc="Exporting model",
    ) as pbar:
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(pbar,),
        )
        monitor_thread.start()

        # process.communicate replies with [stdout, stderr]
        stderr_output: str = process.communicate()[1]  # Wait for completion
        monitor_thread.join()

    if process.returncode != 0:
        error_message = (
            f"CRITICAL: optimum-cli Export failed with return code "
            f"{process.returncode}."
        )
        if stderr_output:
            error_message += f"\nStderr:\n{stderr_output.strip()}"
        raise CLIExportError(error_message)


def quantize_and_save_model(
    input_dir: Path,
    output_dir: Path,
    layer_precision_mapping: dict[str, list[str]] | None = None,
    unknown_quantization_level: str | None = None,
) -> None:
    """Consume an OpenVINO-IR FP16 model and quantize it as specified.

    Args:
        input_dir (Path): The location on disk of the OpenVINO-IR model
                          to be quantized.
        output_dir (Path): The location on disk where the quantized model
                           should be saved.
        layer_precision_mapping (dict[str, list[str]] | None): Mapping of
            quantization levels to lists of layer name substrings.
        unknown_quantization_level (str | None): Default quantization level
            for layers not matching the mapping.
    """
    # Set defaults for optional parameters to avoid mutable default arguments
    # and ensure the function can be called with minimal parameters if desired.
    if layer_precision_mapping is None:
        layer_precision_mapping = {}
    if unknown_quantization_level is None:
        unknown_quantization_level = ""

    # The config generated by optimum-cli export may contain `loss_type=None`,
    # which is not recognized by OVModelForCausalLM and causes a warning.
    # We load the config, remove the attribute, and pass the cleaned config
    # to suppress this harmless warning.
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
    if hasattr(config, "loss_type"):
        delattr(config, "loss_type")

    # Load a valid FP16 OpenVINO-IR.
    model: OptimizedModel = OVModelForCausalLM.from_pretrained(
        input_dir, config=config, export=False, compile=False
    )

    # ov.Model is a subclass of PreTrainedModel, but to access the underlying
    # OpenVINO model, we need to set the type of model.model to ov.Model.
    ov_model: ov.Model = model.model  # type: ignore

    sorting_result = sort_layers_by_precision(
        ov_model, layer_precision_mapping, unknown_quantization_level
    )
    param_counts = sorting_result.param_counts
    layer_map = sorting_result.layer_map

    for current_qlevel in layer_map.keys():
        # The model already became entirely F16 during conversion.
        # Quant levels with 0 parameters can be skipped.
        if current_qlevel == "FP16" or param_counts[current_qlevel] == 0:
            continue
        ignored_names: list[str] = []
        # Select layers by process of elimination using the full layer names.
        for qlevel, layers in layer_map.items():
            if qlevel == current_qlevel:
                continue
            for layer in layers:
                ignored_names.append(layer)

        # NNCF's INT8 modes do not support group_size, while INT4 modes do.
        # We set it conditionally.
        compress_weights_kwargs = {}
        if "INT4" in current_qlevel:
            compress_weights_kwargs["group_size"] = 128

        mode: nncf.CompressWeightsMode = QLEVEL_STR_TO_NNCF_MODE.get(
            current_qlevel,
            nncf.CompressWeightsMode.INT8_SYM,
        )

        # The model is compressed and returned. We update model.model with the
        # new compressed model in each iteration.
        model.model = nncf.compress_weights(
            model.model,
            mode=mode,
            ignored_scope=(
                nncf.IgnoredScope(names=ignored_names)
                if ignored_names
                else None
            ),
            **compress_weights_kwargs,
        )
        logging.info(
            "Completed %s quantization with %s parameters compressed.",
            current_qlevel,
            param_counts[current_qlevel],
        )

    # Save the final quantized model.
    model.save_pretrained(output_dir)


def run_sanitized_conversion() -> None:
    """Execute the full conversion and quantization pipeline.

    Orchestrates the multi-stage process to convert a HuggingFace FP16 model
    to an OpenVINO-IR quantized model. This includes:
    1. Converting the HuggingFace model to OpenVINO-IR format
    2. Quantizing the model according to the layer precision mapping
    3. Converting and saving the tokenizer in OpenVINO format

    The function cleans up intermediate directories before and after execution,
    and performs a final sanity check on the resulting model file size.

    Note:
        Output directories (final_dir, intermediate_dir) will be overwritten
        if they already exist.
    """
    # FIXME: This function has become a stand-in for __main__.
    # TODO: Explain clearly that output will be overwritten.
    # TODO: Maybe add a prompt to confirm deletion?
    for dir_path in [final_dir, intermediate_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        convert_hf_to_openvino_ir(
            input_dir=fake_fp16_dir, output_dir=intermediate_dir
        )
        quantize_and_save_model(
            input_dir=intermediate_dir,
            output_dir=final_dir,
            layer_precision_mapping=LAYER_PRECISION_MAPPING,
            unknown_quantization_level=UNKNOWN_QUANTIZATION_LEVEL,
        )
        format_and_save_tokenizer(
            input_dir=fake_fp16_dir,
            output_dir=final_dir,
        )
    finally:
        # Cleanup the temp dir.
        if intermediate_dir.exists():
            shutil.rmtree(intermediate_dir)

    # TODO: Remove this crude file size check.
    bin_file = final_dir / "openvino_model.bin"
    if not bin_file.exists():
        logging.error("Quantization failed, model file not created.")
        return
    size_gb = bin_file.stat().st_size / (1024**3)
    print(f"\n[Result] Final Model Size: {size_gb:.2f} GB")


if __name__ == "__main__":
    run_sanitized_conversion()

import shutil
import subprocess
import math
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
import json
import nncf
from typing import NamedTuple
from optimum.intel import OVModelForCausalLM
from optimum.modeling_base import OptimizedModel
from openvino_tokenizers import convert_tokenizer
import openvino as ov
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("nncf").setLevel(logging.DEBUG)

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
    "/media/Sandisk-500G/kubernetes-volumes/llm-quantization/"
    "Cydonia-Temp-Dir"
)
final_dir: Path = Path(
    "/media/Sandisk-500G/kubernetes-volumes/llm-quantization/"
    "Cydonia-Mixed-Quant-RtN"
)
# Here's my thinking to preserve LLM quality while maximizing quantization:
# - Keep the embedding and output layers in FP16, as they are critical for
#   maintaining model quality.
# - Quantize the attention projection layers to INT8_SYM, as they are important
#   but can tolerate some quantization without a significant quality drop.
# - Quantize the feedforward layers to INT4_SYM, as they are less sensitive
#   to quantization and can benefit from the smaller size.
# My rough estimates show that this results in a compressed model that's about
# 10% bigger in VRAM than a full INT4 quantization, but with significantly
# better quality. Ask your favorite LLM to categorize your HuggingFace model's
# layers into these buckets to generate a .json file that maps layer names to
# quantization levels and input that into this API.
LAYER_PRECISION_MAPPING: dict[str, list[str]] = {
    "FP16": ["embed", "lm_head"],
    "INT8_SYM": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "INT4_SYM": ["gate_proj", "up_proj", "down_proj"],
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
    pass


class CLIExportError(Error):
    """Raised when the optimum-cli export command fails."""
    pass


def sort_layers_by_precision(ov_model: ov.Model,
                             layer_precision_map: dict[str, list[str]],
                             unknown_quantization_level: str,
                             ) -> LayerSortingResult:
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
            logging.info(
                f"Skipping non-constant op \"{op.get_friendly_name()}\" "
                f"of type \"{op.get_type_name()}\".")
            continue

        # The 'layer' for NNCF is the operation that *consumes* the constant
        # weight, not the constant itself. We find the consumer node to get
        # the correct name for the ignored_scope.
        target_inputs = op.get_output_target_inputs(0)
        if not target_inputs:
            logging.debug(
                f"Constant op '{op.get_friendly_name()}' has no consumers. "
                "Skipping."
            )
            continue
        # A weight constant is typically consumed by only one operation.
        consumer_op = next(iter(target_inputs)).get_node()
        layer: str = consumer_op.get_friendly_name().lower()

        # Get parameter count (size) of the weight constant
        params: int = math.prod(op.get_output_tensor(0).get_shape())

        # This ternary provides substring matching from user-defined layer
        # names to model layer names.
        # e.g. "q_proj" in "model.layers.0.self_attn.q_proj.weight"
        if any(x in layer for x in layer_precision_map.get("FP16", [])):
            layer_map["FP16"].append(layer)
            param_counts["FP16"] += params
        elif any(x in layer for x in layer_precision_map.get("INT8_SYM", [])):
            layer_map["INT8_SYM"].append(layer)
            param_counts["INT8_SYM"] += params
        elif any(x in layer for x in layer_precision_map.get("INT4_SYM", [])):
            layer_map["INT4_SYM"].append(layer)
            param_counts["INT4_SYM"] += params
        else:
            logging.info(
                f"Unknown quantization level for op: "
                f"\"{op.get_friendly_name()}\" with {params} parameters. "
                f"Assigning to {unknown_quantization_level}.")
            layer_map[unknown_quantization_level].append(layer)
            param_counts[unknown_quantization_level] += params

    # This output is really just for fun.
    print("Parameter Count by Quantization Precision:")
    total_params = sum(param_counts.values())
    for qlevel, qlevel_params in param_counts.items():
        percentage = (
            qlevel_params / total_params) * 100 if total_params > 0 else 0
        print(f"  {qlevel}: {qlevel_params} parameters ({percentage:.2f}%)")
    return LayerSortingResult(param_counts=param_counts, layer_map=layer_map)


def format_and_save_tokenizer(
    input_dir: Path, output_dir: Path,
) -> None:
    # Check if the model is a Mistral model by inspecting its config
    # TODO: "config.json" is specific to HuggingFace.
    # TODO: Save the filename to a variable.
    logging.info("\nChecking model config to determine tokenizer formatting.")
    config_path: Path = input_dir / "config.json"
    is_mistral_model: bool = False
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Cannot determine model type for tokenizer formatting."
        )
    with open(config_path, "r") as f:
        config = json.load(f)
        if config.get("model_type") == "mistral":
            is_mistral_model = True

    # If it's a Mistral model, apply the regex fix while loading.
    # TODO: Do I need to save these to disk before converting to OpenVINO-IR?
    if is_mistral_model:
        logging.info("\nMistral model detected. Applying tokenizer regex fix.")
        tokenizer = AutoTokenizer.from_pretrained(
            input_dir, trust_remote_code=True, fix_mistral_regex=True
        )
    else:
        logging.info("\nModel is not Mistral, skipping tokenizer patch.")
        tokenizer = AutoTokenizer.from_pretrained(
            input_dir, trust_remote_code=True
        )

    try:
        ov_tokenizer: ov.Model | tuple[ov.Model, ov.Model] = convert_tokenizer(
            tokenizer, with_detokenizer=True)
        if isinstance(ov_tokenizer, tuple):
            ov.save_model(ov_tokenizer[0],
                          output_dir / "openvino_tokenizer.xml")
            ov.save_model(ov_tokenizer[1],
                          output_dir / "openvino_detokenizer.xml")
        else:
            ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
    except Exception as e:
        logging.error(f"Tokenizer conversion failure: {e}")


def convert_hf_to_openvino_ir(input_dir: Path, output_dir: Path) -> None:
    """
    Convert the HuggingFace FP16 model into an OpenVINO-IR FP16 model.

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
    # How are the nice "Applying Weight Compression" loading bars done in the
    # optimum-cli tool?
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
    logging.info(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise CLIExportError(f"CRITICAL: optimum-cli Export failed: {e}")


def quantize_and_save_model(input_dir: Path,
                            output_dir: Path,
                            layer_precision_mapping: dict[str, list[str]] = {},
                            unknown_quantization_level: str = "") -> None:
    """Consume an OpenVINO-IR FP16 model and quantize it as specified.

    Args:
        input_dir (Path): The location on disk of the OpenVINO-IR model
                          to be quantized.
        output_dir (Path): The location on disk where the quantized model
                           should be saved.
    """

    # The config generated by optimum-cli export may contain `loss_type=None`,
    # which is not recognized by OVModelForCausalLM and causes a warning.
    # We load the config, remove the attribute, and pass the cleaned config
    # to suppress this harmless warning.
    config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
    if hasattr(config, "loss_type"):
        delattr(config, "loss_type")

    # Load a valid FP16 OpenVINO-IR.
    model: OptimizedModel = OVModelForCausalLM.from_pretrained(
        input_dir,
        config=config,
        export=False,
        compile=False)

    # ov.Model is a subclass of PreTrainedModel, but to access the underlying
    # OpenVINO model, we need to set the type of model.model to ov.Model.
    ov_model: ov.Model = model.model  # type: ignore
    sorting_result = sort_layers_by_precision(
        ov_model, layer_precision_mapping, unknown_quantization_level)
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
                nncf.IgnoredScope(
                    names=ignored_names) if ignored_names else None
            ),
            **compress_weights_kwargs,
        )
        logging.info(
            f"Completed {current_qlevel} quantization with "
            f"{param_counts[current_qlevel]} "
            f"parameters compressed.")

    # Save the final quantized model.
    model.save_pretrained(output_dir)


def run_sanitized_conversion() -> None:
    # FIXME: This function has become a stand-in for __main__.
    # TODO: Explain clearly that output will be overwritten.
    # TODO: Maybe add a prompt to confirm deletion?
    for dir_path in [final_dir, intermediate_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    try:
        convert_hf_to_openvino_ir(input_dir=fake_fp16_dir,
                                  output_dir=intermediate_dir)
    except Exception as e:
        raise e

    try:
        quantize_and_save_model(
            input_dir=intermediate_dir,
            output_dir=final_dir,
            layer_precision_mapping=LAYER_PRECISION_MAPPING,
            unknown_quantization_level=UNKNOWN_QUANTIZATION_LEVEL)
    except Exception as e:
        raise e

    try:
        format_and_save_tokenizer(
            input_dir=fake_fp16_dir,
            output_dir=final_dir)
    except Exception as e:
        raise e

    # Cleanup the temp dir.
    shutil.rmtree(intermediate_dir)

    # TODO: Remove this crude file size check.
    bin_file = final_dir / "openvino_model.bin"
    size_gb = bin_file.stat().st_size / (1024**3)
    print(f"\n[Result] Final Model Size: {size_gb:.2f} GB")

    # TODO: This is specifically for the Cydonia model I'm trying.
    # It should be 12 GB.
    if 10 < size_gb < 14:
        print("SUCCESS: Model is INT4 and likely Stateful.")
        print(f"Ready to serve: {final_dir}")
    else:
        print("WARNING: Model size is unexpected.")


if __name__ == "__main__":
    run_sanitized_conversion()

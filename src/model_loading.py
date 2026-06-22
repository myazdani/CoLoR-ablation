from __future__ import annotations

import sys
from pathlib import Path

import torch


REQUIRED_HF_CHECKPOINT_FILES = (
    "config.json",
    "pytorch_model.bin",
)


def validate_hf_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {path}\n"
            "If this is running in Colab, verify Drive is mounted and "
            "configs/default.yaml points to the converted checkpoint folder "
            "under /content/drive/MyDrive/color-filter-ablation/assets/hf/."
        )
    missing = [name for name in REQUIRED_HF_CHECKPOINT_FILES if not (path / name).is_file()]
    if missing:
        present = sorted(p.name for p in path.iterdir())
        raise FileNotFoundError(
            f"Checkpoint directory is missing converted HF files: {missing}\n"
            f"Directory: {path}\n"
            f"Present files: {present}\n"
            "Expected a converted checkpoint. If this folder only has "
            "model.pt/config.yaml, run scripts/05_convert_olmo_to_hf.py first."
        )
    return path


def register_local_olmo(paper_code_path: str | Path | None) -> None:
    if not paper_code_path:
        return
    path = Path(paper_code_path).resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        from hf_olmo.configuration_olmo import OLMoConfig
        from hf_olmo.modeling_olmo import OLMoForCausalLM
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception as exc:
        raise RuntimeError(f"Could not register local OLMo code from {path}") from exc

    try:
        AutoConfig.register("olmo", OLMoConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(OLMoConfig, OLMoForCausalLM)
    except ValueError:
        pass
    _patch_olmo_output_embeddings(OLMoForCausalLM)


def _patch_olmo_output_embeddings(model_cls: type) -> None:
    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        return self.model.transformer.ff_out_last

    def set_output_embeddings(self, value):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out_last = value

    model_cls.get_output_embeddings = get_output_embeddings
    model_cls.set_output_embeddings = set_output_embeddings
    # Newer Transformers releases consult this dict during meta-device loading.
    # The paper fork predates it and only relies on its no-op tie_weights().
    if not hasattr(model_cls, "all_tied_weights_keys"):
        model_cls.all_tied_weights_keys = {}


def load_causal_lm(
    checkpoint_path: str | Path,
    *,
    paper_code_path: str | Path | None = None,
    dtype: str = "bf16",
):
    register_local_olmo(paper_code_path)
    checkpoint_path = validate_hf_checkpoint_dir(checkpoint_path)

    kwargs = {}
    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif dtype not in {"auto", "none", ""}:
        raise ValueError(f"Unsupported dtype '{dtype}'")

    if paper_code_path:
        from hf_olmo.modeling_olmo import OLMoForCausalLM

        return OLMoForCausalLM.from_pretrained(str(checkpoint_path), **kwargs)

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before loading HF checkpoints") from exc
    kwargs["trust_remote_code"] = True
    return AutoModelForCausalLM.from_pretrained(str(checkpoint_path), **kwargs)

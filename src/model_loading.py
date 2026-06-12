from __future__ import annotations

import sys
from pathlib import Path

import torch


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


def load_causal_lm(
    checkpoint_path: str | Path,
    *,
    paper_code_path: str | Path | None = None,
    dtype: str = "bf16",
):
    register_local_olmo(paper_code_path)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before loading HF checkpoints") from exc

    kwargs = {"trust_remote_code": True}
    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype == "fp32":
        kwargs["torch_dtype"] = torch.float32
    elif dtype not in {"auto", "none", ""}:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return AutoModelForCausalLM.from_pretrained(str(checkpoint_path), **kwargs)


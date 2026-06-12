#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def write_config(checkpoint_dir: Path) -> None:
    from hf_olmo.configuration_olmo import OLMoConfig
    from olmo import ModelConfig

    model_config = ModelConfig.load(checkpoint_dir / "config.yaml", key="model")
    config_kwargs = model_config.asdict()
    config_kwargs["use_cache"] = True
    config = OLMoConfig(**config_kwargs)
    config.save_pretrained(checkpoint_dir)


def write_model(checkpoint_dir: Path, *, remove_olmo_files: bool) -> None:
    from hf_olmo.modeling_olmo import OLMoForCausalLM

    old_model_path = checkpoint_dir / "model.pt"
    new_model_path = checkpoint_dir / "pytorch_model.bin"
    state_dict = torch.load(old_model_path, map_location=torch.device("cpu"))
    new_state_dict = {
        f"{OLMoForCausalLM.base_model_prefix}.{key}": value
        for key, value in state_dict.items()
    }
    torch.save(new_state_dict, new_model_path)
    if remove_olmo_files:
        old_model_path.unlink()


def write_tokenizer(checkpoint_dir: Path) -> None:
    from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast
    from olmo import Tokenizer

    tokenizer_raw = Tokenizer.from_checkpoint(checkpoint_dir)
    tokenizer = OLMoTokenizerFast(
        tokenizer_object=tokenizer_raw.base_tokenizer,
        truncation=tokenizer_raw.truncate_direction,
        max_length=tokenizer_raw.truncate_to,
        eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False),
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.pad_token_id = tokenizer_raw.pad_token_id
    tokenizer.eos_token_id = tokenizer_raw.eos_token_id
    tokenizer.save_pretrained(checkpoint_dir)


def fix_bad_tokenizer(checkpoint_dir: Path) -> None:
    from omegaconf import OmegaConf as om

    config_path = checkpoint_dir / "config.yaml"
    conf = om.load(config_path)
    conf["tokenizer"]["identifier"] = "allenai/gpt-neox-olmo-dolma-v1_5"
    conf["model"]["eos_token_id"] = 50279
    om.save(conf, config_path)


def convert_checkpoint(checkpoint_dir: Path, *, remove_olmo_files: bool) -> None:
    fix_bad_tokenizer(checkpoint_dir)
    write_config(checkpoint_dir)
    write_model(checkpoint_dir, remove_olmo_files=remove_olmo_files)
    write_tokenizer(checkpoint_dir)
    if remove_olmo_files:
        os.remove(checkpoint_dir / "config.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an OLMo checkpoint to HF format using CPU-safe torch.load."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--remove-olmo-files",
        action="store_true",
        help="Remove model.pt/config.yaml after conversion. Not recommended for local debugging.",
    )
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    assets_dir = checkpoint_dir.parent.parent
    default_cache = assets_dir / ".cached_path"
    default_hf_home = assets_dir / ".hf-cache"
    os.environ.setdefault("CACHED_PATH_CACHE_ROOT", str(default_cache))
    os.environ.setdefault("HF_HOME", str(default_hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(default_hf_home / "hub"))
    Path(os.environ["CACHED_PATH_CACHE_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HUGGINGFACE_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    convert_checkpoint(checkpoint_dir, remove_olmo_files=args.remove_olmo_files)
    print(f"Converted {checkpoint_dir}")


if __name__ == "__main__":
    main()

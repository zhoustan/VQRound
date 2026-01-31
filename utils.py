import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import transformers


def hash_ids(x: torch.Tensor) -> str:
    return hashlib.md5(x.detach().to("cpu", dtype=torch.int32).numpy().tobytes()).hexdigest()


def find_linears_in_layer(layer: nn.Module) -> Dict[str, nn.Module]:
    out = {}
    for n, m in layer.named_modules():
        if isinstance(m, (nn.Linear, transformers.Conv1D, nn.Conv2d)):
            out[n] = m
    return out


def get_parent(root: nn.Module, path: str):
    parts = path.split(".")
    cur = root
    for p in parts[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        elif isinstance(cur, (nn.ModuleDict, dict)) and p in cur:
            cur = cur[p]
        else:
            cur = getattr(cur, p)
    return cur, parts[-1]


def set_child(parent: nn.Module, child: str, new_mod: nn.Module):
    if child.isdigit():
        parent[int(child)] = new_mod
    elif isinstance(parent, (nn.ModuleDict, dict)) and child in parent:
        parent[child] = new_mod
    else:
        setattr(parent, child, new_mod)


def collect_trainable_codebooks(model: nn.Module):
    params, mods = [], []
    from vqround import VQRoundLinear
    for m in model.modules():
        if isinstance(m, VQRoundLinear):
            m.codebook.data = m.codebook.data.float()
            m.codebook.requires_grad_(True)
            params.append(m.codebook)
            mods.append(m)
    return params, mods


def freeze_stochastic_but_keep_train(model: nn.Module):
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for k in [
            "dropout", "attention_dropout", "activation_dropout",
            "hidden_dropout", "hidden_dropout_prob",
            "attention_dropout_prob", "attention_probs_dropout_prob",
            "embd_pdrop", "resid_pdrop", "attn_pdrop",
            "classifier_dropout", "summary_first_dropout",
            "ffn_dropout", "ff_dropout", "dropout_rate",
            "layerdrop", "encoder_layerdrop", "decoder_layerdrop",
            "drop_path_rate",
        ]:
            if hasattr(cfg, k):
                setattr(cfg, k, 0.0)

    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = 0.0
        if m.__class__.__name__ in {"StableDropout", "T5Dropout"}:
            if hasattr(m, "dropout_prob"):
                m.dropout_prob = 0.0
            if hasattr(m, "p"):
                m.p = 0.0
        if m.__class__.__name__ in {"StochasticDepth", "DropPath"}:
            if hasattr(m, "p"):
                m.p = 0.0
            if hasattr(m, "drop_prob"):
                m.drop_prob = 0.0
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
        for attr in ["attention_dropout", "attn_dropout", "resid_dropout",
                     "hidden_dropout", "dropout_p", "dropout"]:
            if hasattr(m, attr) and isinstance(getattr(m, attr), float):
                setattr(m, attr, 0.0)

    model.train(True)
    if hasattr(model, "config"):
        model.config.use_cache = False


def set_quant_mode(model: nn.Module, mode: str):
    from vqround import VQRoundLinear
    for m in model.modules():
        if isinstance(m, VQRoundLinear):
            m.set_quant_mode(mode)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_attn_implementation(prefer_flash: bool = True) -> str:
    if not prefer_flash:
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"

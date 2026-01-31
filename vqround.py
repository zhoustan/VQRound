import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

from quant import Quantizer
from gptq import GPTQ

from utils import find_linears_in_layer, get_parent, set_child

class RectifiedSigmoid(nn.Module):
    def __init__(self, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.gamma = gamma
        self.zeta = zeta

    def forward(self, x):
        return torch.clamp(torch.sigmoid(x) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def inverse(self, y):
        return -torch.log((self.zeta - self.gamma) / (y - self.gamma) - 1)

def per_row_affine_params(w: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    out, in_dim = w.shape
    wmin = w.amin(dim=1, keepdim=True)
    wmax = w.amax(dim=1, keepdim=True)
    qmax = (1 << n_bits) - 1
    scale = (wmax - wmin) / max(1, qmax)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.round(-wmin / scale)
    return scale.view(-1, 1), zero.view(-1, 1)



class VQRoundLinear(nn.Module):
    def __init__(
            self,
            base_linear: nn.Linear,
            n_bits: int = 4,
            D: int = 8,
            K: int = 2 ** 12,
            kmeans_iters: int = 30,
            gamma: float = -0.1,
            zeta: float = 1.1,
            external_scale: torch.Tensor = None,
            external_zero: torch.Tensor = None,
            external_base_int: torch.Tensor = None,
            external_rest: torch.Tensor = None,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach().clone())
        else:
            self.bias = None

        device = base_linear.weight.device
        w_dtype = base_linear.weight.dtype
        self.w_dtype = w_dtype

        self.n_bits = int(n_bits)

        if external_scale is not None and external_zero is not None and external_base_int is not None:
            self.register_buffer("scale", external_scale.to(device=device, dtype=w_dtype))
            self.register_buffer("zero", external_zero.to(device=device, dtype=w_dtype))

            self.register_buffer("base_int", external_base_int.to(device=device, dtype=torch.int16))
            rest = external_rest.to(device=device, dtype=w_dtype)
            rest = torch.clamp(rest, 0, 1)
        else:
            W = base_linear.weight.data.detach().clone()
            s, z = per_row_affine_params(W.float(), n_bits)
            scale = s.to(device=device, dtype=W.dtype)
            zero = z.to(device=device, dtype=W.dtype)
            self.register_buffer("scale", scale.to(device=device, dtype=w_dtype))
            self.register_buffer("zero", zero.to(device=device, dtype=w_dtype))
            self.register_buffer("base_int", torch.floor(W.float() / scale).to(device=device, dtype=torch.int16))
            rest = (W.float() / scale) - torch.floor(W.float() / scale)
            rest = torch.clamp(rest, 0, 1)

        rsig = RectifiedSigmoid(gamma, zeta).to(device)
        self.rsig = rsig

        alpha_full = rsig.inverse(rest)
        base_shape = alpha_full.shape
        n = alpha_full.numel()
        L = (n + D - 1) // D
        pad = L * D - n
        if pad:
            alpha_vec = torch.cat([alpha_full.reshape(-1), torch.zeros(pad, device=device, dtype=alpha_full.dtype)],
                                  dim=0)
        else:
            alpha_vec = alpha_full.reshape(-1)
        alpha_blocks = alpha_vec.view(L, D)

        with torch.no_grad():
            codebook_init, indices = self._kmeans(alpha_blocks, K=K, iters=kmeans_iters)

        self.register_buffer("indices", indices.to(torch.int16))
        self.D = int(D)
        self.base_shape = base_shape
        self.n = n
        self.codebook = nn.Parameter(codebook_init.to(device=device, dtype=torch.float16).clone())
        self.quant_mode = "soft"

    def set_quant_mode(self, mode: str):
        assert mode in ("soft", "hard")
        self.quant_mode = mode

    def _reconstruct_alpha(self) -> torch.Tensor:
        alpha_blocks = self.codebook[self.indices.int()]
        alpha_flat = alpha_blocks.view(-1)[: self.n]
        return alpha_flat.view(self.base_shape)

    def _r(self, alpha: torch.Tensor) -> torch.Tensor:
        return (alpha >= 0).float() if self.quant_mode == "hard" else self.rsig(alpha)

    @torch.no_grad()
    def _kmeans(self, X: torch.Tensor, K: int, iters: int = 30):
        kmeans = faiss.Kmeans(X.shape[1], K, niter=iters, verbose=False, gpu=True)
        kmeans.train(X.cpu())
        centroids = torch.from_numpy(kmeans.centroids).to(X.device)
        idx = torch.from_numpy(kmeans.index.search(X.cpu(), 1)[1][:, 0]).to(X.device)
        return centroids, idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._reconstruct_alpha()
        r = self._r(alpha)
        base_int_f = self.base_int.to(dtype=self.scale.dtype)
        q_int = torch.clamp(base_int_f + r + self.zero, 0, 2 ** self.n_bits - 1)
        Wq = self.scale * (q_int - self.zero)
        Wq = Wq.to(self.w_dtype)
        return F.linear(x, Wq, self.bias)


@torch.no_grad()
def gptq_sequential_collect_grid_opt(
        model: AutoModelForCausalLM,
        calib_loader: DataLoader,
        device: torch.device,
        wbits: int = 4,
        groupsize: int = 128,
        sym: bool = True,
        actorder: bool = False,
        percdamp: float = 0.01,
        static_groups: bool = False,
        blocksize: int = 128,
        write_quantized_weights: bool = False,
) -> Dict[str, dict]:
    dev = device
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if not hasattr(model, 'model'):  # BLOOM
        model_type = 'bloom'
        layers = model.transformer.h
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    else:
        if hasattr(model.model, 'decoder'):  # OPT
            model_type = "opt"
            layers = model.model.decoder.layers
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        else:
            model_type = "llama"
            layers = model.model.layers
            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            if getattr(model.model, "norm", None) is not None:
                model.model.norm = model.model.norm.to(dev)
            if hasattr(model.model, "rotary_emb"):
                model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype

    try:
        nsamples = len(calib_loader)
        batched = False
    except TypeError:
        calib_batches = list(calib_loader)
        nsamples = len(calib_batches)
        batched = True

    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if model_type == 'llama':
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    if not batched:
        for batch in calib_loader:
            ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            try:
                model(ids.to(dev))
            except ValueError:
                pass
    else:
        for batch in calib_batches:
            ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            try:
                model(ids.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    if model_type == 'llama':
        position_ids = cache['position_ids']

    grids: Dict[str, dict] = {}

    for li in range(len(layers)):
        layer = layers[li].to(dev)
        subset = find_linears_in_layer(layer)

        gptqs: Dict[str, GPTQ] = {}
        origW: Dict[str, torch.Tensor] = {}

        for name, mod in subset.items():
            g = GPTQ(mod)
            g.quantizer = Quantizer()
            g.quantizer.configure(bits=wbits, perchannel=True, sym=sym, mse=False)
            gptqs[name] = g
            origW[name] = mod.weight.data.detach().clone()

        handles = []

        def add_batch(name):
            def _hook(_mod, inp, out):
                gptqs[name].add_batch(inp[0].data, out.data)

            return _hook

        for name, mod in subset.items():
            handles.append(mod.register_forward_hook(add_batch(name)))

        with torch.inference_mode():
            for j in range(nsamples):
                if model_type == 'llama':
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask
                    )[0]

        for h in handles: h.remove()

        for name, g in gptqs.items():
            print(name)
            ret = g.fasterquant_return_with_error(
                blocksize=blocksize, percdamp=percdamp,
                groupsize=groupsize, actorder=actorder, static_groups=static_groups
            )
            # wq_fp = ret["Q"].detach().cpu().to(torch.float16).clone()
            if hasattr(model.model, 'decoder'):  # opt
                key = f"model.decoder.layers.{li}.{name}"
            else:  # llama
                key = f"model.layers.{li}.{name}"

            grids[key] = {
                "scale": ret["scale"].cpu(),
                "zero": ret["zero"].cpu(),
                "maxq": ret["maxq"],
                "groupsize": groupsize,
                "sym": sym,
                # "wq": wq_fp,
                "base_int": ret["base_int"].cpu(),
                "rest": ret["rest"].cpu(),
            }

            if not write_quantized_weights:
                subset[name].weight.data.copy_(origW[name].to(subset[name].weight.dtype))
            g.free()

        del gptqs, origW

        with torch.inference_mode():
            for j in range(nsamples):
                if model_type == 'llama':
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask
                    )[0]

        layers[li] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if not hasattr(model, 'model'):  # BLOOM
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    else:
        if hasattr(model.model, 'decoder'):  # OPT
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
            if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
                model.model.decoder.project_out = model.model.decoder.project_out.cpu()
            if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        else:  # LLaMA
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            if getattr(model.model, "norm", None) is not None:
                model.model.norm = model.model.norm.cpu()
            if hasattr(model.model, "rotary_emb"):
                model.model.rotary_emb = model.model.rotary_emb.cpu()
    model.config.use_cache = use_cache
    return grids



def replace_linear_with_vqround_using_grids(
        model: nn.Module,
        grids: dict,
        n_bits=4, D=8, K=4096, kmeans_iters=30,
        skip_keywords: Optional[List[str]] = None
) -> int:
    if skip_keywords is None:
        skip_keywords = ["lm_head", "embed_tokens", "embed_positions"]

    to_replace = []

    if hasattr(model, "model") and hasattr(model.model, "decoder"):  # OPT
        layer_list = model.model.decoder.layers
        layer_prefix = "model.decoder.layers"
    else:  # LLaMA
        layer_list = model.model.layers
        layer_prefix = "model.layers"

    for li, layer in enumerate(layer_list):
        for subname, submod in layer.named_modules():
            if not isinstance(submod, nn.Linear):
                continue
            full_name = f"{layer_prefix}.{li}.{subname}"
            if any(kw in full_name for kw in (skip_keywords or [])):
                continue
            to_replace.append((full_name, submod))

    def get_parent(root: nn.Module, path: str):
        parts = path.split(".");
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

    cnt = 0
    for name, lin in to_replace:
        print(name, flush=True)
        grid = grids.get(name, None)
        parent, child = get_parent(model, name)
        if grid is not None:
            new_m = VQRoundLinear(
                base_linear=lin, n_bits=n_bits, D=D, K=K, kmeans_iters=kmeans_iters,
                external_scale=grid["scale"], external_zero=grid["zero"],
                # external_wq=grid.get("wq", None),
                external_base_int=grid["base_int"], external_rest=grid["rest"],
            )
        else:
            new_m = VQRoundLinear(base_linear=lin, n_bits=n_bits, D=D, K=K, kmeans_iters=kmeans_iters, )
        set_child(parent, child, new_m)
        cnt += 1
    return cnt


def harden_and_export(student: nn.Module, save_dir: str):
    wrappers = []
    for name, m in student.named_modules():
        if isinstance(m, VQRoundLinear):
            wrappers.append((name, m))

    def get_parent(root: nn.Module, path: str):
        parts = path.split(".");
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

    with torch.no_grad():
        for name, m in wrappers:
            alpha = m._reconstruct_alpha()
            r_hard = (alpha >= 0).float()
            base_int = m.base_int.to(dtype=m.scale.dtype)
            q_int = torch.clamp(base_int + r_hard + m.zero, 0, 2 ** m.n_bits - 1)
            Wq = m.scale * (q_int - m.zero)
            Wq = Wq.to(m.w_dtype)

            new_lin = nn.Linear(m.in_features, m.out_features, bias=(m.bias is not None),
                                device=Wq.device, dtype=Wq.dtype)
            new_lin.weight.copy_(Wq)
            if m.bias is not None:
                new_lin.bias.copy_(m.bias.data)

            parent, child = get_parent(student, name)
            set_child(parent, child, new_lin)

    n_left = sum(1 for _ in student.modules() if isinstance(_, VQRoundLinear))
    print(f"[export] VQRoundLinear remaining: {n_left}")
    assert n_left == 0, "Some VQRoundLinear remain after harden!"

    os.makedirs(save_dir, exist_ok=True)
    student.save_pretrained(save_dir)
    print(f"[export] Saved hardened model to {save_dir}")

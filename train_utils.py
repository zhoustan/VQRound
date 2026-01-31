import os
import time
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from utils import hash_ids, collect_trainable_codebooks
from vqround import VQRoundLinear


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float, start_b: float, end_b: float):
        self.t_max = int(t_max)
        self.start_decay = rel_start_decay * self.t_max
        self.start_b = float(start_b)
        self.end_b = float(end_b)

    def __call__(self, t: int) -> float:
        if t < self.start_decay: return self.start_b
        rel_t = (t - self.start_decay) / max(1.0, (self.t_max - self.start_decay))
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1.0 - rel_t)


class RoundLoss(nn.Module):
    def __init__(self, max_count: int, b_range: Tuple[float, float], decay_start: float, warmup: float, p_norm: float):
        super().__init__()
        self.loss_start = max_count * warmup
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.p_norm = p_norm
        self.b = 0.0

    def forward(self, iter_count: int, sb: torch.Tensor, override_b: Optional[float] = None) -> torch.Tensor:
        if iter_count < self.loss_start: return sb.new_zeros(())
        self.b = float(override_b) if override_b is not None else self.temp_decay(iter_count)
        return (1.0 - (2.0 * sb - 1.0).abs().pow(self.b)).sum()


def _kd_loss_with_temperature(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                              T: float = 1.0) -> torch.Tensor:
    if T == 1.0:
        log_p_s = F.log_softmax(student_logits, dim=-1)
        log_p_t = F.log_softmax(teacher_logits, dim=-1)
        return F.kl_div(log_p_s, log_p_t, log_target=True, reduction="batchmean")
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    log_p_t = F.log_softmax(teacher_logits / T, dim=-1)
    return F.kl_div(log_p_s, log_p_t, log_target=True, reduction="batchmean") * (T * T)


def _round_reg_checkpointed(m, beta_now, step, rnd_loss, amp_dtype):
    def _fn(codebook: torch.Tensor):
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            alpha_blocks = codebook[m.indices.int()]
            alpha_flat = alpha_blocks.view(-1)[: m.n]
            r_soft = m.rsig(alpha_flat)
            return rnd_loss(step, r_soft, override_b=beta_now)

    return checkpoint(_fn, m.codebook, use_reentrant=False)


@torch.no_grad()
def build_teacher_cache(calib_data, teacher, device, T: float, save_dir: str):
    teacher.eval()
    items = []
    for idx, pair in enumerate(calib_data):
        inp = pair[0] if isinstance(pair, (list, tuple)) else pair
        input_ids = inp.to(device)
        logits = teacher(input_ids).logits
        if T != 1.0: logits = logits / T
        logp = F.log_softmax(logits, dim=-1).to(torch.float16).cpu()
        h = hash_ids(inp)
        items.append({"h": h, "logp": logp})
        if (idx + 1) % 10 == 0:
            print(f"[cache-onefile] {idx + 1}/{len(calib_data)} collected")
    index = {it["h"]: i for i, it in enumerate(items)}
    blob = {"type": "full_logp_single", "T": float(T), "items": items, "index": index}
    torch.save(blob, save_dir)
    print(f"[cache-onefile] saved {len(items)} items to single file: {save_dir}")


def collect_trainable_codebooks(model: nn.Module) -> Tuple[List[nn.Parameter], List[VQRoundLinear]]:
    params, mods = [], []
    for m in model.modules():
        if isinstance(m, VQRoundLinear):
            m.codebook.data = m.codebook.data.float()
            m.codebook.requires_grad_(True)
            params.append(m.codebook)
            mods.append(m)
    return params, mods


def freeze_stochastic_but_keep_train(model):
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
            if hasattr(cfg, k): setattr(cfg, k, 0.0)
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = 0.0
        if m.__class__.__name__ in {"StableDropout", "T5Dropout"}:
            if hasattr(m, "dropout_prob"): m.dropout_prob = 0.0
            if hasattr(m, "p"): m.p = 0.0
        if m.__class__.__name__ in {"StochasticDepth", "DropPath"}:
            if hasattr(m, "p"): m.p = 0.0
            if hasattr(m, "drop_prob"): m.drop_prob = 0.0
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
        for attr in ["attention_dropout", "attn_dropout", "resid_dropout",
                     "hidden_dropout", "dropout_p", "dropout"]:
            if hasattr(m, attr) and isinstance(getattr(m, attr), float):
                setattr(m, attr, 0.0)
    model.train(True)
    if hasattr(model, "config"):
        model.config.use_cache = False


def train_e2e_kd(
        student: nn.Module,
        teacher: Optional[nn.Module],
        train_loader: DataLoader,
        steps: int,
        lr: float,
        device: torch.device,
        kd_T: float = 2.0,
        kd_alpha: float = 1.0,
        use_round_reg: bool = True,
        beta_hi: float = 10.0,
        beta_lo: float = 2.0,
        beta_hold_ratio: float = 0.1,
        round_weight: float = 0.01,
        log_interval: int = 50,
        use_kd: bool = True,
        teacher_cache: Optional[str] = None,
        verify_cache: bool = True
):
    # def _sync_time():
    #     torch.cuda.synchronize(device) if torch.cuda.is_available() else None
    #     return time.time()

    student.to(device)
    for p in student.parameters():
        p.requires_grad_(False)
    train_params, mods = collect_trainable_codebooks(student)
    assert len(train_params) > 0, "No VQ codebooks to train!"

    # for name, param in student.named_parameters():
    #     print(f"  [param] {name} | shape = {param.shape} | requires_grad = {param.requires_grad}")
    optimizer = torch.optim.Adam(train_params, lr=lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    rnd_loss = RoundLoss(steps, b_range=(beta_hi, beta_lo), decay_start=0.0, warmup=0.1, p_norm=2.0)

    total_cb = sum(m.codebook.numel() for m in mods)
    print(f"[train] VQRoundLinear modules: {len(mods)} | total codebook params: {total_cb}")

    cache_items, cache_index, T_cache = None, None, 2.0
    if teacher_cache is not None:
        cache_blob = torch.load(teacher_cache, map_location="cpu")
        assert cache_blob.get("type") in ("full_logp_single", "full_logp_memmap"), "wrong cache type"
        T_cache = float(cache_blob.get("T", 2.0))
        if cache_blob["type"] == "full_logp_single":
            cache_items = cache_blob["items"]
            cache_index = cache_blob.get("index", None)
        print(
            f"[cache] loaded T={T_cache}; items={len(cache_items) if cache_items is not None else 'memmap'}; index={'ok' if cache_index else 'missing'}")

    data_iter = iter(train_loader)
    step, best = 0, float("inf")

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch[0].to(device, non_blocking=True) if isinstance(batch, (list, tuple)) else batch.to(device,
                                                                                                             non_blocking=True)

        if step <= steps * beta_hold_ratio:
            beta_now = beta_hi
        else:
            beta_now = beta_hi - (beta_hi - beta_lo) * (step / steps)
        amp_dtype = (
            torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            if cache_items is not None:
                s_logits = student(input_ids).logits

                cur_ids = batch[0] if isinstance(batch, (list, tuple)) else batch
                h = hash_ids(cur_ids)
                if cache_index is not None and h in cache_index:
                    item = cache_items[cache_index[h]]
                    t_logp_cpu = item["logp"]
                else:
                    if step == 0:
                        print("[cache] no index or hash miss; falling back to sequential pointer (results may differ).")
                    item = cache_items[step % len(cache_items)]
                    t_logp_cpu = item["logp"] if isinstance(item, dict) else item
                if verify_cache and isinstance(item, dict) and "h" in item:
                    assert item["h"] == h, f"[cache] misaligned at step={step}"

                T_s, T_t = s_logits.size(1), t_logp_cpu.size(1)
                Tm = min(T_s, T_t)
                s_logp_T = F.log_softmax(s_logits[:, -Tm:, :] / T_cache, dim=-1)
                t_logp = t_logp_cpu[:, -Tm:, :].to(device, dtype=s_logp_T.dtype, non_blocking=True)
                task_loss = F.kl_div(s_logp_T, t_logp, log_target=True, reduction="batchmean") * (T_cache * T_cache)
                task_loss = task_loss * kd_alpha
            else:
                if teacher is not None and use_kd:
                    with torch.no_grad():
                        t_logits = teacher(input_ids).logits
                    s_logits = student(input_ids).logits
                    task_loss = _kd_loss_with_temperature(s_logits, t_logits, T=kd_T) * kd_alpha
                else:
                    outputs = student(input_ids, labels=input_ids)
                    task_loss = outputs.loss

        if use_round_reg:
            rr = 0.0
            for m in mods:
                rr = rr + _round_reg_checkpointed(
                    m=m, beta_now=beta_now, step=step, rnd_loss=rnd_loss, amp_dtype=amp_dtype
                )
            total_loss = task_loss + round_weight * rr
        else:
            rr = torch.tensor(0.0, device=device)
            total_loss = task_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1
        if step % log_interval == 0:
            print(
                f"[e2e] step {step}/{steps} | beta={beta_now:.2f} | task={task_loss.item():.4f} | round={rr.item():.4f}",
                flush=True)
            best = min(best, float(task_loss.item()))
    return best

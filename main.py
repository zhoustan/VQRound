import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import get_weight_quant_data
from eval_utils import evaluate

from utils import set_seed, freeze_stochastic_but_keep_train, set_quant_mode
from vqround import gptq_sequential_collect_grid_opt, replace_linear_with_vqround_using_grids, harden_and_export
from train_utils import build_teacher_cache, train_e2e_kd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--calib_data", type=str, default="c4", choices=["c4", "wikitext2"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--gptq_only", action="store_true", help="GPTQ baseline")
    parser.add_argument("--pre_gptq", action="store_true", help="Enable GPTQ")
    parser.add_argument("--gptq_blocksize", type=int, default=128)
    parser.add_argument("--gptq_percdamp", type=float, default=0.01)
    parser.add_argument("--gptq_groupsize", type=int, default=-1)
    parser.add_argument("--gptq_sym", action="store_true", default=False)
    parser.add_argument("--gptq_actorder", action="store_true", default=False)
    parser.add_argument("--gptq_static_groups", action="store_true", default=False)

    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--kmeans_iters", type=int, default=500)
    parser.add_argument("--skip_keywords", nargs="*", default=["lm_head"])

    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=1.0)
    parser.add_argument("--use_round_reg", action="store_true", default=True)
    parser.add_argument("--round_weight", type=float, default=0.01)
    parser.add_argument("--beta_hi", type=float, default=20.0)
    parser.add_argument("--beta_lo", type=float, default=2.0)
    parser.add_argument("--beta_hold_ratio", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="kd", choices=["kd", "lm"])

    parser.add_argument("--build_teacher_cache", action="store_true", default=True)
    parser.add_argument("--teacher_cache", type=str, default=None)

    parser.add_argument("--export_dir", type=str, default=None)

    args = parser.parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    args.model_name = args.model_path.split('/')[-1]
    args.model_type = args.model_name.split('-')[0]
    print(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_name = args.teacher_model if args.teacher_model else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    tokenizer.pad_token = tokenizer.eos_token

    calib_data = get_weight_quant_data(args)

    teacher = None
    teacher_cache_path = args.teacher_cache
    use_kd = (args.loss == "kd") and (args.teacher_model is not None)
    if use_kd:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
        for p in teacher.parameters(): p.requires_grad = False
        if args.build_teacher_cache:
            if teacher_cache_path is None:
                teacher_cache_path = os.path.join(
                    args.cache_dir,
                    f"teacher_full_logp_{args.model_type}_{args.calib_data}_ns{args.nsamples}_L{args.seqlen}_T{args.kd_temperature}.pt"
                )
            if os.path.exists(teacher_cache_path):
                print(f"[cache-onefile] teacher_cache {teacher_cache_path} already exists, skip building.")
            else:
                print(f"[cache-onefile] building teacher_cache to {teacher_cache_path} ...")
                build_teacher_cache(calib_data, teacher, device=device, T=args.kd_temperature,
                                    save_dir=teacher_cache_path)
            del teacher
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            teacher = None

    student = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2",
    )
    student.seqlen = args.seqlen
    if args.gptq_only:
        grids = gptq_sequential_collect_grid_opt(
            student, calib_loader=calib_data, device=device,
            wbits=args.w_bits,
            groupsize=args.gptq_groupsize if args.gptq_groupsize > 0 else -1,
            sym=args.gptq_sym,
            actorder=args.gptq_actorder,
            percdamp=args.gptq_percdamp,
            static_groups=args.gptq_static_groups,
            blocksize=args.gptq_blocksize,
            write_quantized_weights=True,
        )

        evaluate(student, args)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_dir = args.export_dir
        if not exp_dir or str(exp_dir).strip() == "":
            exp_dir = os.path.join(
                "exports",
                f"{args.model_name}"
                f"_gptq_only"
                f"_gptq_w{args.w_bits}"
                f"_g{args.gptq_groupsize if args.gptq_groupsize > 0 else 'full'}"
                f"_b{args.gptq_blocksize}_d{args.gptq_percdamp}"
                f"_sym{'Y' if args.gptq_sym else 'N'}_act{'Y' if args.gptq_actorder else 'N'}"
                f"_ns{args.nsamples}_L{args.seqlen}"
                f"_{ts}"
            )
        os.makedirs(exp_dir, exist_ok=True)
        student.save_pretrained(exp_dir)
        print(f"[export] GPTQ-only model saved to: {exp_dir}")
        exit()
        return

    save_path = os.path.join("cache", f"gptq_grids_{args.model_path.split('/')[1]}_wbits_{args.w_bits}.pt")
    grids = {}
    if args.pre_gptq:
        if os.path.exists(save_path):
            print(f"[gptq] loading pre-collected GPTQ grids from {save_path}")
            grids = torch.load(save_path, map_location="cpu")
        else:
            print(f"[gptq] collecting GPTQ grids ...")
            grids = gptq_sequential_collect_grid_opt(
                student, calib_loader=calib_data, device=device,
                wbits=args.w_bits,
                groupsize=args.gptq_groupsize if args.gptq_groupsize > 0 else -1,
                sym=args.gptq_sym,
                actorder=args.gptq_actorder,
                percdamp=args.gptq_percdamp,
                static_groups=args.gptq_static_groups,
                blocksize=args.gptq_blocksize,
                write_quantized_weights=False,
            )
            torch.save(grids, save_path)

    n_replaced = replace_linear_with_vqround_using_grids(
        student, grids, n_bits=args.w_bits, D=args.D, K=args.K,
        kmeans_iters=args.kmeans_iters, skip_keywords=args.skip_keywords
    )
    print(f"[wrap] replaced {n_replaced} Linear -> VQRoundLinear (GPTQ grid)")

    del grids
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:
        evaluate(student, args)
    except Exception as e:
        print(f"[eval] failed: {e}")

    freeze_stochastic_but_keep_train(student)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    import time
    t0 = time.time()
    train_e2e_kd(
        student=student,
        teacher=None,
        train_loader=calib_data,
        steps=args.steps,
        lr=args.lr,
        device=device,
        kd_T=args.kd_temperature,
        kd_alpha=args.kd_alpha,
        use_round_reg=args.use_round_reg,
        beta_hi=args.beta_hi,
        beta_lo=args.beta_lo,
        beta_hold_ratio=args.beta_hold_ratio,
        round_weight=args.round_weight,
        log_interval=50,
        use_kd=(args.loss == "kd"),
        teacher_cache=teacher_cache_path,
        verify_cache=True
    )
    t1 = time.time()
    print(f"[train] time cost: {t1 - t0:.1f} seconds")

    student.gradient_checkpointing_disable()
    set_quant_mode(student, "soft")
    try:
        evaluate(student, args)
    except Exception as e:
        print(f"[eval] failed: {e}")

    set_quant_mode(student, "hard")
    try:
        evaluate(student, args)
    except Exception as e:
        print(f"[eval] failed: {e}")

    exp_dir = args.export_dir
    if not exp_dir or str(exp_dir).strip() == "":
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.pre_gptq:
            exp_dir = os.path.join(
                "exports",
                f"{args.model_name}"
                f"_w{args.w_bits}_D{args.D}_K{args.K}"
                f"_ns{args.nsamples}_L{args.seqlen}"
                f"_lr{args.lr}_bs{args.batch_size}"
                f"_T{args.kd_temperature}_steps{args.steps}"
                f"_gptqGrid_sym{'Y' if args.gptq_sym else 'N'}_g{args.gptq_groupsize if args.gptq_groupsize > 0 else 'full'}"
                f"_act{'Y' if args.gptq_actorder else 'N'}"
                f"_{ts}"
            )
        else:
            exp_dir = os.path.join(
                "exports",
                f"{args.model_name}"
                f"_w{args.w_bits}_D{args.D}_K{args.K}"
                f"_ns{args.nsamples}_L{args.seqlen}"
                f"_lr{args.lr}_bs{args.batch_size}"
                f"_T{args.kd_temperature}_steps{args.steps}"
                f"_{ts}"
            )
    print(f"[export] export_dir = {exp_dir}")
    harden_and_export(student, exp_dir)


if __name__ == "__main__":
    main()

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import get_weight_quant_data_with_tokenizer
from eval_utils import evaluate
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--load_path", type=str, required=True,
                        help="export_dir from your script (save_pretrained output)")
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--calib_data", type=str, default="wikitext2", choices=["c4", "wikitext2"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    args.model_name = args.model_path.split('/')[-1]
    args.model_type = args.model_name.split('-')[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.load_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map=None,
    ).to(device)

    model.seqlen = args.seqlen

    _ = get_weight_quant_data_with_tokenizer(args, tokenizer)
    evaluate(model, args)


if __name__ == "__main__":
    main()

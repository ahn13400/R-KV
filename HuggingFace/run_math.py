import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3

dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_new_tokens = {
    "gsm8k": 8192,
    "aime24": 32768,
    "math": 16384,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def main(args):
    fout = open(args.save_path, "w")

    prompts = []
    test_data = []

    with open(args.dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[args.dataset_name][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)
            if args.apply_chat_template:
                # R1-distill models are trained to reason inside the chat
                # format (<think> after the assistant turn); a plain-text
                # prompt makes them exit thinking immediately and lowers
                # accuracy substantially.
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)

    # Subset sampling for quick testing. `test_data` is deliberately left at full
    # length: it is only ever indexed by sample_idx, which is bounded by len(prompts).
    prompts = prompts[: int(args.fraction * len(prompts))]

    # Print some samples for sanity check
    for prompt in prompts[:5]:
        print("#" * 100)
        print(prompt)
        print("#" * 100)


    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i : i + args.eval_batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            # the chat template already inserts BOS/special tokens
            add_special_tokens=not args.apply_chat_template,
        ).to("cuda")

        prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()
        prompt_sequence_length = tokenized_prompts.input_ids.shape[1]

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "num_beams": 1,
            "num_return_sequences": args.num_return_sequences,
        }
        if args.do_sample:
            generation_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )

        output = model.generate(
            **tokenized_prompts,
            **generation_kwargs,
        )

        batch_token_stats = []
        for j in range(output.size(0)):
            prompt_idx = j // args.num_return_sequences
            prefill = prefill_lengths[prompt_idx]
            generated_ids = output[j][prompt_sequence_length:]
            output_tokens = int((generated_ids != tokenizer.pad_token_id).sum().item())
            total_tokens = prefill + output_tokens

            batch_token_stats.append(
                {
                    "sample_idx": i + prompt_idx,
                    "candidate_idx": j % args.num_return_sequences,
                    "prefill_tokens": prefill,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

        batch_outputs = [
            tokenizer.decode(output[j][prompt_sequence_length:], skip_special_tokens=True)
            for j in range(output.size(0))
        ]

        torch.cuda.empty_cache()

        for j in range(len(batch_outputs)):
            sample_idx = batch_token_stats[j]["sample_idx"]
            prompt_idx = sample_idx - i
            result = dict(test_data[sample_idx])
            result["prompt"] = batch_prompts[prompt_idx]
            result["output"] = batch_outputs[j]
            result["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
            result["output_tokens"] = batch_token_stats[j]["output_tokens"]
            result["total_tokens"] = batch_token_stats[j]["total_tokens"]
            result["sample_idx"] = batch_token_stats[j]["sample_idx"]
            result["candidate_idx"] = batch_token_stats[j]["candidate_idx"]

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    fout.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o", "covariance_merge", "rkv_merge", "rkv_merge_anchor"],
    )
    parser.add_argument("--kv_budget", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=1.0,
        help="covariance_merge only: merge a source into its nearest target only if the predicted "
        "variance of their attention-logit gap, scaling^2 * dk^T Sigma dk, is <= this. Units are "
        "squared logits, so sqrt is a predicted std deviation of the gap in nats.",
    )
    parser.add_argument("--update_kv", type=bool, default=True)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )

    # model config
    parser.add_argument(
        "--divide_method",
        type=str,
        default="generated_length",
        choices=["newline", "step_length", "generated_length"],
        help="generated_length mod divide length is the compression clock. This is batch-agnostic"
    )
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument(
        "--ema_half_life",
        type=int,
        default=64,
        help="covariance_merge only: half-life H (in steps) of the EMA over query moments; the "
        "decay is 2^(-1/H). H=64 was best in the qstats estimator sweep.",
    )
    parser.add_argument(
        "--future_horizon",
        type=int,
        default=128,
        help="covariance_merge only: number of future RoPE frames P the query moments are averaged "
        "over. P=128 was best in the qstats estimator sweep.",
    )
    parser.add_argument(
        "--future_decay",
        type=float,
        default=1.0,
        help="covariance_merge only: weight on future offset h is future_decay^(h-1), normalized. "
        "1.0 is uniform and is the default -- recency weighting only hurt in the sweep.",
    )
    parser.add_argument(
        "--compression_content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="enable sampling; paper-style reproduction uses this with temperature 0.6 and top_p 0.95",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="wrap the prompt with tokenizer.apply_chat_template; recommended "
        "for DeepSeek-R1-distill models, which only enter their trained "
        "<think> reasoning format inside the chat template",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="evaluate only the first fraction of the dataset, for quick testing",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="passed through to apply_chat_template as enable_thinking=; only takes "
        "effect together with --apply_chat_template, and only for models whose "
        "template reads it (e.g. Qwen3)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.num_return_sequences > 1 and not args.do_sample:
        raise ValueError("--num_return_sequences > 1 requires --do_sample")
    set_seed(args.seed)

    args.dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    if args.max_new_tokens == -1: args.max_new_tokens = dataset2max_new_tokens[args.dataset_name]

    # ====== build compression config ======
    compression_config = {
        "method": args.method,
        "method_config": {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
            "merge_threshold": args.merge_threshold,
        },
        "compression": None,
        "update_kv": args.update_kv
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
        # read by CausalLM_forward / the attention layers for covariance_merge
        "ema_half_life": args.ema_half_life,
        "future_horizon": args.future_horizon,
        "future_decay": args.future_decay,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # apply monkey patch
    if args.method.lower() != "fullkv":
        if "llama" in args.model_path.lower():
            replace_llama(compression_config)
        elif "qwen3" in args.model_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in args.model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    model.config.update(model_config)

    if args.method.lower() != "fullkv":
        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]

        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]

    main(args)

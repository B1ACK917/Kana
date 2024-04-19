import torch
import time
import json
import argparse

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
)
from utils.bench import bench_inference

# args
parser = argparse.ArgumentParser("Kana IPEX Benchmark Kit", add_help=False)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="model path",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16"],
    default="bfloat16",
    help="bfloat16, float32",
)
parser.add_argument("--max-new-tokens", default=32, type=int, help="output max new tokens")
parser.add_argument("--bench", default="inference", type=str, help="select from [inference, finetune, all]")
parser.add_argument("--peft", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--num-iter", default=20, type=int, help="num iter")
parser.add_argument("--num-warmup", default=2, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
args = parser.parse_args()
print(args)

# dtype
amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
config = AutoConfig.from_pretrained(
    args.model, trust_remote_code=True
)
if not hasattr(config, "lm_head_generation"):
    config.lm_head_generation = True

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=amp_dtype,
    config=config,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
tokenizer = LlamaTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = model.eval()
model = model.to(memory_format=torch.channels_last)

# to ipex
if args.ipex:
    import intel_extension_for_pytorch as ipex

    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        print("ipex._C.disable_jit_linear_repack() failed")

    model = ipex.llm.optimize(
        model.eval(),
        dtype=amp_dtype,
        inplace=True,
    )

# generate args
generate_kwargs = dict(do_sample=False, max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens)


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


if __name__ == '__main__':
    # input prompt
    with open("./prompt.json") as f:
        prompt_pool = json.load(f)
    prompt = prompt_pool["llama"]["32"]
    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    # set bench
    valid = ["inference", "finetune"]
    if args.bench == "all":
        bench = valid
    else:
        bench = [args.bench]
    for b in bench:
        assert b in valid, f"{b} not in f{valid}"

    # start
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=amp_enabled
    ):
        for bench_type in bench:
            match bench_type:
                case "inference":
                    total_time = bench_inference(tokenizer, model, prompt, generate_kwargs, num_iter, num_warmup)
                case "finetune":
                    total_time = bench_inference(tokenizer, model, prompt, generate_kwargs, num_iter, num_warmup)
                case _:
                    total_time = 0
            print("\n", "-" * 10, "Summary:", "-" * 10)
            latency = total_time / (num_iter - num_warmup)
            print(f"{bench_type} latency: {latency:.3f} sec.")

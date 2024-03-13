import torch
import time
import json
import argparse
import numpy as np
from itertools import chain

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
)

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
parser.add_argument("--profile", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--torch-compile", action="store_true")
parser.add_argument("--backend", default="ipex", type=str, help="backend of torch.compile")
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

if args.torch_compile:
    model.forward = torch.compile(model.forward, dynamic=True, backend=args.backend)

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

    # start
    total_time = 0.0
    num_iter = args.num_iter
    num_warmup = args.num_warmup
    prompt = [prompt] * args.batch_size
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=amp_enabled
    ):
        # profile
        if args.profile:
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
                    on_trace_ready=trace_handler,
            ) as prof:
                for i in range(5):
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    output = model.generate(input_ids, **generate_kwargs)
                    prof.step()

        # benchmark
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = model.generate(input_ids, **generate_kwargs)
            gen_ids = output
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o - i
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            if i >= num_warmup:
                total_time += toc - tic

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)
    print("Inference latency: %.3f sec." % latency)

import torch
import json
import argparse

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
)
from utils.bench import bench_inference, bench_finetune
from utils.dataset import DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN, DEFAULT_PAD_TOKEN
from peft import LoraConfig

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
parser.add_argument("--epoch", default=1.0, type=float, help="train epoch")
parser.add_argument("--output", default="temp/model", type=str, help="finetune model output dir")
parser.add_argument("--data", default="data/alpaca_small.json", type=str, help="finetune data input")
parser.add_argument("--device", default="cpu", help="map device")
args = parser.parse_args()
print(args)


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


if __name__ == '__main__':
    # dtype
    amp_enabled = True if args.dtype != "float32" else False
    amp_dtype = getattr(torch, args.dtype)

    # load model
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if not hasattr(config, "lm_head_generation"):
        config.lm_head_generation = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=amp_dtype,
        config=config,
        device_map=args.device
    )
    if args.peft:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.add_adapter(peft_config)
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

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
    for bench_type in bench:
        match bench_type:
            case "inference":
                model = model.eval()
                model = model.to(memory_format=torch.channels_last)
                with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
                    total_time = bench_inference(tokenizer, model, prompt, generate_kwargs, num_iter, num_warmup)
                print("\n", "-" * 10, "Summary:", "-" * 10)
                latency = total_time / (num_iter - num_warmup)
                print(f"{bench_type} latency: {latency:.3f} sec.")
            case "finetune":
                training_args = TrainingArguments(output_dir=args.output, num_train_epochs=args.epoch,
                                                  per_device_train_batch_size=args.batch_size,
                                                  use_cpu=True if args.device == "cpu" else False)
                total_time, result = bench_finetune(args.data, tokenizer, model, training_args)
                print("\n", "-" * 10, "Summary:", "-" * 10)
                print(f"{bench_type} total elapsed: {total_time:.3f} sec.")
            case _:
                total_time = 0
                print("No valid --bench, skip")

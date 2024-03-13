from transformers import AutoConfig, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser("Kana Benchmark Model Parameter Generator", add_help=False)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="model path (should contain config.json)",
)
args = parser.parse_args()

config = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_config(config=config)
model.save_pretrained(args.model)

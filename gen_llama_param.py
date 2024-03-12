from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

config = AutoConfig.from_pretrained('./model/Llama-2-7b-hf')
model = AutoModelForCausalLM.from_config(config=config)
tokenizer = AutoTokenizer.from_pretrained('./model/Llama-2-7b-hf')
model.save_pretrained("model/test")
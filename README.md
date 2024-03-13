# Kana -- IPEX Pytorch LLM Benchmark kit

## 1. Startup
``` bash
# Clone repository
git clone https://github.com/B1ACK917/Kana.git
cd Kana

# Init submodule
git submodule sync
git submodule update --init --recursive
```

## 2.1 Docker based build
```bash
# Build docker image
make image

# Run container and step into it
make run

# Step into a running container
# docker run --it kana:main bash
```

## 2.2 Conda based build
```bash
# Create conda env
conda create -n kana python=3.10 -y
conda activate kana

# Use ipex env
cd thirdparty/intel-extension-for-pytorch/examples/cpu/inference/python/llm
bash ./tools/env_setup.sh 7

# Activate environment variables
# If this makes bug, try not activate environment variables
source ./tools/env_activate.sh
```

## 3 Benchmark
### 3.1 Generate Llama parameters
Clone a full version of Llama parameter is very time consuming, so I only storage the model config and token parameters. Thus you should use `gen_llama_param.py` to generate a random version of Llama parameter to run the benchmark.
**Note this generated parameters may produce weired output, but we only want the inferencetime, so we don't care about the output.**
```bash
python gen_llama_param.py -m model/Llama-2-7b-chat-hf
```
### 3.2 Running with pytorch
```bash
OMP_NUM_THREADS=$(nproc)$ python bench_llama.py -m model/Llama-2-7b-hf --dtype float32
```

### 3.3 Running with IPEX accelerated pytorch
```bash
OMP_NUM_THREADS=$(nproc)$ python bench_llama.py -m model/Llama-2-7b-hf --dtype float32 --ipex
```

### 3.4 Running with IPEX accelerated pytorch and with a bfloat16 instead of float32
```bash
OMP_NUM_THREADS=$(nproc)$ python bench_llama.py -m model/Llama-2-7b-hf --dtype bfloat16 --ipex
```
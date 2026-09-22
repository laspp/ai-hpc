# Running a simple chat model inside a container

The example uses the `transformers` library to download a chat model from the Hugging Face Hub and run a single query against it.
It is packaged as an Apptainer container.

## Contents

```
01_llm_query/
├── query_model.py                       # the script: loads a model and answers one prompt
├── requirements.txt                     # Python dependencies
├── containers/
│   ├── local_llm.def                    # downloads the model at run time
│   └── local_qwen.def                   # bakes the model Qwen2.5-0.5B-Instruct into the image at build time
└── README.md
```

## Building and using the container with a preloaded model

 Prebuilt container can be pulled from `ghcr.io/laspp/local_qwen:latest`:

```bash
apptainer pull oras://ghcr.io/laspp/local_qwen:latest
```

If you want to build the container yourself use the definition file `local_qwen.def`. Note, that the definition file's `%files` entries (`../query_model.py`, `../requirements.txt`) are relative to the directory you run `apptainer build` from, so build it from inside `containers/`. By default `Qwen/Qwen2.5-0.5B-Instruct` is downloaded and baked into the image. To build the model:

```bash
cd containers
srun --cpus-per-taks=16 apptainer build local_qwen_latest.sif local_qwen.def
```

You can change the model using `--build-arg`:

```bash
srun --cpus-per-taks=16 apptainer build --build-arg MODEL=microsoft/Phi-3-mini-4k-instruct \
    local_phi.sif local_qwen.def
```

The above commands build the container on a cluster compute node. Ommit `srun --cpus-per-taks=16` if building locally.

Run the model on the Arnes cluster:

```bash
# CPU
srun apptainer run local_qwen_latest.sif \
    --prompt "What is the capital of Slovenia?"

# GPU
srun --partition=gpu --gpus=1 apptainer run --nv local_qwen_latest.sif \
    --prompt "Explain KV-cache in one sentence." --max-new-tokens 64
```

## Building and using the container with no preloaded model

Prebuilt container can be pulled from `ghcr.io/laspp/local_llm:latest`:

```bash
apptainer pull oras://ghcr.io/laspp/local_llm:latest
```

If you want to build the container yourself use the definition file `local_llm.def`. Note, that the definition file's `%files` entries (`../query_model.py`, `../requirements.txt`) are relative to the directory you run `apptainer build` from, so build it from inside `containers/`: The definition file's `%files` entries (`../query_model.py`,`../requirements.txt`) are relative to the directory you run `apptainer build` from, so build it from inside `containers/`:

```bash
cd containers
srun --cpus-per-taks=16 apptainer build local_llm_latest.sif local_llm.def
```
Create a Hugging Face cache directory to store the downloaded models in your home folder:

```bash
mkdir -p ~/hf-cache
```

Run the model on the Arnes cluster:
```bash
# CPU
srun apptainer run --bind ~/hf-cache:/opt/app/hf-cache \
    local_llm_latest.sif --prompt "What is the capital of Slovenia?"

# GPU
srun --partition=gpu --gpus=1 apptainer run --nv --bind ~/hf-cache:/opt/app/hf-cache \
    local_llm_latest.sif \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "Explain KV-cache in one sentence." \
    --max-new-tokens 64
```

## Run the Python script with no container

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch
python query_model.py --prompt "What is the capital of Slovenia?"
```

## Python script options

```
--model            Hugging Face repo id (default: Qwen/Qwen2.5-0.5B-Instruct)
--prompt           the user prompt
--max-new-tokens   generation length (default: 128)
--temperature      sampling temperature; 0 = greedy decoding (default: 0.7)
```

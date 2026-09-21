# Private LLM server using vLLM and Open WebUI

Serve a Hugging Face model via [vLLM](https://github.com/vllm-project/vllm)'s OpenAI-compatible API, with [Open WebUI](https://github.com/open-webui/open-webui)
as a browser-based chat frontend, packaged as a single Apptainer container. 

## Contents

```
03_llm_server/
├── containers/
│   └── vllm_server.def       # Apptainer definition file
└── run_vllm_openwebui_server.sbatch # SLURM submission script
```

## Building the container

Prebuilt container can be pulled from `ghcr.io/laspp/llm_server:latest`:

```bash
apptainer pull oras://ghcr.io/laspp/llm_server:latest
```

To build it yourself, run:

```bash
cd containers
apptainer build llm_server_latest.sif llm_server.def
```

## Submitting as a SLURM job

Edit the `# Configuration` block at the top of [run_vllm_openwebui_server.sbatch](run_vllm_openwebui_server.sbatch) (model,
ports, API key, ...) if needed. The script is to be submitted to the Arnes cluster. To run the server:

```bash
sbatch run_vllm_openwebui_server.sbatch
```
You need to wait for the services to come online. Check the log `vllm_openwebui_server.log` for when the information `Application startup complete.` appears.
The job's log  also prints the SSH tunnel command for the node it landed on at the beginning, for example:

```bash
ssh -N -L 8080:<node>:8080 -L 8000:<node>:8000 <user>@hpc-login3.arnes.si
```
Run the above command from your own computer, then open `http://127.0.0.1:8080` in a
browser.

## Configuration options

These are te options you can configure inside the `run_vllm_openwebui_server.sbatch` script:
```
MODEL_ID                 HF repo id to serve (default: Qwen/Qwen2.5-0.5B-Instruct)
SERVED_MODEL_NAME        Name exposed via the API / shown in OpenWebUI (update this too when you change MODEL_ID)
VLLM_PORT                vLLM's listening port (default: 8000)
OPENWEBUI_PORT           OpenWebUI's listening port (default: 8080)
LOGIN_HOST               Login node used in the printed SSH tunnel command
TENSOR_PARALLEL_SIZE     GPUs to split the model across (from --gpus-per-node)
MAX_MODEL_LEN            Context length cap, in tokens (default: 4096)
GPU_MEMORY_UTILIZATION   Fraction of GPU memory vLLM may reserve (default: 0.9)
API_KEY                  Shared API key for both services (change the default!)
```

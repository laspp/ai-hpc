# LLM server with vLLM + OpenWebUI

Serve a Hugging Face model via [vLLM](https://github.com/vllm-project/vllm)'s
OpenAI-compatible API, with [OpenWebUI](https://github.com/open-webui/open-webui)
as a browser-based chat frontend, packaged as a single Apptainer container
for the Arnes HPC cluster. Access is via an SSH tunnel from your own
computer to the compute node.

## Contents

```
03_llm_server/
├── containers/
│   └── vllm_server_recipe.def       # Apptainer definition file
└── run_vllm_openwebui_server.sbatch # SLURM submission script
```

vLLM and OpenWebUI are installed into two separate Python virtual
environments inside the same image.

## Building the container


```bash
cd containers
apptainer build llm_server_latest.sif llm_server.def
```

## Submitting as a SLURM job

Edit the `# Configuration` block at the top of
[run_vllm_openwebui_server.sbatch](run_vllm_openwebui_server.sbatch) (model,
ports, API key, ...) if needed, then submit it from `03_llm_server/`:

```bash
sbatch run_vllm_openwebui_server.sbatch
```

The job's log (`vllm_openwebui_server.log`) prints the SSH tunnel
command for the node it landed on:

```bash
ssh -N -L 8080:<node>:8080 -L 8000:<node>:8000 <user>@hpc-login3.arnes.si
```

Run the above command from your own computer, then open `http://127.0.0.1:8080` in a
browser.

## Configuration options

```
MODEL_ID                 HF repo id to serve (default: Qwen/Qwen2.5-0.5B-Instruct)
SERVED_MODEL_NAME         name exposed via the API / shown in OpenWebUI
                          (update this too when you change MODEL_ID)
VLLM_PORT                 vLLM's listening port (default: 8000)
OPENWEBUI_PORT             OpenWebUI's listening port (default: 8080)
LOGIN_HOST                 login node used in the printed SSH tunnel command
TENSOR_PARALLEL_SIZE       GPUs to split the model across (from --gpus-per-node)
MAX_MODEL_LEN              context length cap, in tokens (default: 4096)
GPU_MEMORY_UTILIZATION     fraction of GPU memory vLLM may reserve (default: 0.9)
API_KEY                    shared API key for both services (change the default!)
```

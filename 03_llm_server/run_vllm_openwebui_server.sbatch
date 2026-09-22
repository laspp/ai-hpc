#!/bin/bash
#SBATCH --job-name=vllm_openwebui_server
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --constraint=v100s
#SBATCH --exclusive
#SBATCH --output=vllm_openwebui_server.log

# Configuration
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
SERVED_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
VLLM_PORT=8000
OPENWEBUI_PORT=8080
LOGIN_HOST=hpc-login3.arnes.si
TENSOR_PARALLEL_SIZE=${SLURM_GPUS_PER_NODE:-1}
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9
API_KEY=openwebui

WORK_DIR=$(pwd)
CONTAINER_SIF=$WORK_DIR/containers/llm_server_latest.sif

# Persistent host storage for OpenWebUI and HF cache.
OPENWEBUI_DATA_DIR=${OPENWEBUI_DATA_DIR:-$WORK_DIR/data/openwebui}
HF_CACHE_DIR=${HF_CACHE_DIR:-$WORK_DIR/data/hf-cache}

mkdir -p "$OPENWEBUI_DATA_DIR/static" "$HF_CACHE_DIR"

# Empty if no Hugging Face token
HF_TOKEN=""

echo "# starting at $(date)"
echo "# running on ${SLURMD_NODENAME}"
echo "# model: ${MODEL_ID}"
echo "# vLLM port: ${VLLM_PORT}"
echo "# OpenWebUI port: ${OPENWEBUI_PORT}"
echo "# OpenWebUI data dir (host): ${OPENWEBUI_DATA_DIR}"
echo "# HF cache dir (host): ${HF_CACHE_DIR}"
echo
echo "# From your computer, create SSH tunnels with:"
echo "ssh -N -L ${OPENWEBUI_PORT}:${SLURMD_NODENAME}:${OPENWEBUI_PORT} -L ${VLLM_PORT}:${SLURMD_NODENAME}:${VLLM_PORT} ${USER}@${LOGIN_HOST}"
echo
echo "# Then open OpenWebUI: http://localhost:${OPENWEBUI_PORT}"
echo "# API key configured for both services: ${API_KEY}"
echo


echo "# launching vLLM at 0.0.0.0:${VLLM_PORT}"
srun --overlap apptainer exec \
  --nv \
  --bind ${HF_CACHE_DIR}:/hf-cache \
  --env VLLM_USE_V1=0 \
  --env VLLM_ATTENTION_BACKEND=XFORMERS \
  --env HF_TOKEN="${HF_TOKEN}" \
  ${CONTAINER_SIF} \
  /opt/venv/bin/vllm serve "${MODEL_ID}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --api-key "${API_KEY}" \
    --disable-log-stats &

echo "# launching OpenWebUI at 0.0.0.0:${OPENWEBUI_PORT}"
srun --overlap apptainer exec \
  --bind ${OPENWEBUI_DATA_DIR}:/openwebui-data \
  --bind ${HF_CACHE_DIR}:/hf-cache \
  --env DATA_DIR=/openwebui-data \
  --env STATIC_DIR=/openwebui-data/static \
  --env ENABLE_PERSISTENT_CONFIG=False \
  --env DEFAULT_MODEL_PARAMS='{"function_calling":"legacy"}' \
  --env ENABLE_OLLAMA_API=false \
  --env ENABLE_OPENAI_API=true \
  --env OPENAI_API_BASE_URLS="http://localhost:${VLLM_PORT}/v1" \
  --env OPENAI_API_KEYS="${API_KEY}" \
  ${CONTAINER_SIF} \
  /opt/openwebui-venv/bin/open-webui serve --host 0.0.0.0 --port "${OPENWEBUI_PORT}" &

wait -n

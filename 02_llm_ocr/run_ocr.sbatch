#!/bin/bash
#SBATCH --job-name=glm-ocr
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=v100s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=glm-ocr-%j.out


SIF=./containers/local_llm_ocr_latest.sif
HF_CACHE=./hf-cache
PDF_IN=./data/Poezije.pdf     
TXT_OUT=output.txt
echo "Running OCR on $PDF_IN, output to $TXT_OUT"
mkdir -p "$HF_CACHE"

apptainer run --nv --bind "$HF_CACHE:/opt/app/hf-cache" "$SIF" "$PDF_IN" -o "$TXT_OUT" --model "zai-org/GLM-OCR" --prompt "Text Recognition:" --pages "8-11"
    #--bind "$(dirname "$TXT_OUT")":/data_out \
    
#!/bin/bash
#SBATCH --job-name=glm-ocr-array
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=v100s
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=glm-ocr-array-%A_%a.out
#SBATCH --array=0-3

# Multi-GPU version of run_ocr.sbatch: splits START_PAGE-END_PAGE below into
# as many contiguous chunks as there are array tasks, and OCRs each chunk on
# its own GPU in parallel, into its own ocr_parts/part-NNN.txt.
#
# Usage: set START_PAGE/END_PAGE to the page range to process, adjust
# --array=0-N above to (desired GPU count - 1) (e.g. --array=0-3 for 4
# GPUs), then `sbatch run_ocr_array.sbatch`. Once all tasks finish, merge
# the parts yourself with:
#   cat ocr_parts/part-*.txt > output.txt

set -e

SIF=./containers/local_llm_ocr_latest.sif
HF_CACHE=./hf-cache
PDF_IN=./data/Poezije.pdf
OUT_DIR=./ocr_parts
START_PAGE=8
END_PAGE=20

mkdir -p "$OUT_DIR" "$HF_CACHE"

NUM_PAGES=$(( END_PAGE - START_PAGE + 1 ))

# Split the configured page range into NUM_TASKS contiguous, near-equal chunks
START=$(( START_PAGE + (NUM_PAGES * SLURM_ARRAY_TASK_ID) / SLURM_ARRAY_TASK_COUNT ))
END=$(( START_PAGE + (NUM_PAGES * (SLURM_ARRAY_TASK_ID + 1)) / SLURM_ARRAY_TASK_COUNT - 1 ))
PART_OUT="$OUT_DIR/part-$(printf '%03d' "$SLURM_ARRAY_TASK_ID").txt"


echo "Task $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT: OCR-ing pages $START-$END of $PDF_IN, output to $PART_OUT"
apptainer run --nv --bind "$HF_CACHE:/opt/app/hf-cache" "$SIF" \
    "$PDF_IN" -o "$PART_OUT" \
    --model "zai-org/GLM-OCR" --prompt "Text Recognition:" \
    --pages "${START}-${END}"

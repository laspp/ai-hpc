# llm-ocr

Extract text from a PDF using a vision-language OCR model via `transformers`.
Packaged as an Apptainer container for the Arnes HPC cluster.

Defaults to [GaMS-Beta/SVILA-1-4B](https://huggingface.co/GaMS-Beta/SVILA-1-4B),
a Gemma-3 model fine-tuned specifically for **Slovenian** vision-language
tasks (incl. OCR). [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
(a lightweight 0.9B dedicated OCR model) is also supported via `--model`, but
it is **not** Slovenian-tuned — it officially covers only Chinese, English,
French, Spanish, Russian, German, Japanese and Korean, and performs best on
Chinese/English. For Slovenian documents, stick with the SVILA default.

## Contents

```
llm_ocr/
├── ocr_pdf.py        # renders each PDF page and runs the OCR model on it
├── requirements.txt  # Python dependencies (torch comes from the base image)
├── model.def          # Apptainer definition file
├── run_ocr.sbatch     # example SLURM submission script
└── README.md
```

Each PDF page is rasterized with PyMuPDF, sent through the model, and the
per-page results are concatenated into one output text file.

## Option A — run locally, no container

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt torch     # add torch since there's no base image here
python ocr_pdf.py input.pdf -o output.txt
```

## Option B — build and run with Apptainer

```bash
# Build the image (downloads the ~2 GB PyTorch/CUDA base image once)
apptainer build hf.sif model.def

# Run on a GPU node (--nv exposes the GPU to the container)
apptainer run --nv hf.sif input.pdf -o output.txt

# Run on CPU only (slow — fine for a quick test on a login node with a small PDF)
apptainer run hf.sif input.pdf -o output.txt
```

The first run downloads the model from the Hugging Face Hub into `$HF_HOME`
(`/opt/app/hf-cache` inside the container) and caches it. Persist that cache
across runs, and expose your PDF's directory, via bind mounts:

```bash
mkdir -p hf-cache
apptainer run --nv \
    --bind hf-cache:/opt/app/hf-cache \
    --bind /d/hpc/projects/FRI/dsluga/data:/data \
    hf.sif /data/input.pdf -o /data/output.txt
```

## On Arnes: submit as a SLURM job

Never run inference directly on a login node. Submit via `sbatch`, adjusting
`--partition`/`--gpus`/`--account` in [run_ocr.sbatch](run_ocr.sbatch) to match your allocation
(check with `sinfo` / `sacctmgr show associations user=$USER`):

```bash
sbatch run_ocr.sbatch /d/hpc/projects/FRI/dsluga/data/input.pdf \
                       /d/hpc/projects/FRI/dsluga/data/output.txt
```

## Script options

```
--model              HF repo id (default: GaMS-Beta/SVILA-1-4B; try zai-org/GLM-OCR
                      for non-Slovenian documents)
--prompt             task instruction. Default is a Slovenian transcription
                      instruction for SVILA; for GLM-OCR use e.g.
                      "Text Recognition:", "Table Recognition:", "Formula Recognition:"
--pages               page range, 1-indexed, e.g. "1-3,5" (default: all)
--dpi                 page rasterization DPI (default: 150; raise for dense/small text)
--max-long-side       cap the longer side of each rendered page, in pixels
                      (default: 1280; overrides --dpi if it would exceed this —
                      see OOM note below)
--max-new-tokens      generation length per page (default: 4096; use ~8192 for dense pages)
--dtype               bfloat16 (default) | auto | float16 | float32
--attn-implementation e.g. "kernels-community/flash-attn2" if flash-attn is installed
```

## Notes

- `transformers>=5.1.0` is required (needed for `GlmOcrForConditionalGeneration`
  to be registered; SVILA's `Gemma3ForConditionalGeneration` has been available
  for longer). If `AutoModelForImageTextToText.from_pretrained(...)` fails to
  resolve a model, the installed version is too old — install from source
  instead: `pip install -U git+https://github.com/huggingface/transformers.git`.
- Neither SVILA-1-4B nor GLM-OCR is gated — no `HF_TOKEN` needed for the
  defaults here. If you point `--model` at a gated repo (e.g. some Llama or
  Gemma base checkpoints), export `HF_TOKEN` in `%environment` or the job env.
- For scanned/handwritten or very dense documents, raise `--dpi` (e.g. 300)
  and `--max-new-tokens` (e.g. 8192) so pages aren't truncated.
- Flash Attention 2 is not installed in the base container; add
  `pip install -U flash-attn --no-build-isolation` to `%post` in `model.def`
  if you want it (needs a matching CUDA toolchain and takes a while to build).
- SVILA also comes in a 12B variant (`GaMS-Beta/SVILA-1-12B`) for higher
  accuracy at the cost of more GPU memory and slower inference.

### CUDA out of memory in the vision encoder

If you see `torch.OutOfMemoryError` inside `.../modeling_*.py` → `self.visual(...)`
→ `scaled_dot_product_attention`, it's not a model-size problem — it's an
image-resolution problem. Both GLM-OCR's and SVILA's vision towers run full
(non-windowed) attention over every image patch, so memory scales with the
**square** of the page resolution. On a V100 (compute capability 7.0),
PyTorch also can't use the fused/flash SDPA backends here, so it falls back
to a "math" kernel that fully materializes the attention matrix — making
this much easier to hit than on newer GPUs.
`--max-long-side` (default 1280px) exists specifically to bound this; if it
still OOMs, lower it further (e.g. 1024) before reaching for a bigger GPU.

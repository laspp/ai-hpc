# OCR using a Large Language Model

Extract text from a PDF using a vision-language OCR model via `transformers`,
packaged as an Apptainer container for the Arnes HPC cluster.

Defaults to [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), a
lightweight dedicated OCR model — best on Chinese/English, also covers
French/Spanish/Russian/German/Japanese/Korean.

## Contents

```
02_llm_ocr/
├── llm-ocr.py                    # Python script which does the OCR
├── requirements.txt              # Python dependencies
├── containers/
│   └── local_llm_ocr.def         # Apptainer definition file
├── run_ocr.sbatch                # single-GPU SLURM submission example
├── run_ocr_array.sbatch          # multi-GPU (job array) submission example
└── README.md
```

Each selected page is rasterized with PyMuPDF, and all pages are sent
through the model together as a single batch, then the per-page results are
concatenated into one output text file.

## Building the container

Note that the definition file's `%files` entries (`../llm-ocr.py`,
`../requirements.txt`) are relative to the directory you run
`apptainer build` from, so build it from inside `containers/`:

```bash
cd containers
apptainer build local_llm_ocr_latest.sif local_llm_ocr.def
```

## Submitting as a SLURM job

To run the OCR on the Arnes cluster do:

```bash
sbatch run_ocr.sbatch
```

You can change the parameters like input, output, pages in the `run_ocr.sbatch` file


## Multi-GPU: splitting pages across GPUs with a job array

For a large PDF, [run_ocr_array.sbatch](run_ocr_array.sbatch) splits a fixed
page range into as many contiguous, near-equal chunks as there are array
tasks, and OCRs each chunk on its own GPU in parallel:

1. Edit `START_PAGE`/`END_PAGE` and `--array=0-N` (N = GPU count - 1) at the top of the file.
2. Submit it: `sbatch run_ocr_array.sbatch`. Each task writes its own `ocr_parts/part-NNN.txt`.
3. Once every task has finished (check with `squeue`), merge the parts yourself, in order:
   ```bash
   cat ocr_parts/part-*.txt > output.txt
   ```


## Python script options

```
pdf                the input PDF file (positional)
-o, --output       output text file (default: <pdf>.txt)
--model            HF repo id (default: zai-org/GLM-OCR)
--prompt           task instruction (default: "Text Recognition:")
--dpi              page rasterization DPI (default: 150; raise for dense/small text)
--pages            page range, 1-indexed, e.g. "1-3,5" (default: all pages)
--max-new-tokens   generation length per page (default: 4096; raise for dense pages)
--page-separator   text inserted between pages in the output
                    (default: "\n\n----- page {page} -----\n\n")
```

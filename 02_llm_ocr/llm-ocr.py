#!/usr/bin/env python3
"""Extract text from a PDF using a vision-language OCR model. 
    The deafult model used is zai-org/GLM-OCR, which is a general-purpose OCR model.

Usage:
    python llm-ocr.py input.pdf
    python llm-ocr.py input.pdf -o output.txt --pages "1-3,5"
    python llm-ocr.py input.pdf --model zai-org/GLM-OCR \\
        --prompt "Text Recognition:"
    python llm-ocr.py input.pdf --prompt "Table Recognition:" --dpi 300
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import pymupdf
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_page_spec(spec, num_pages):
    if not spec:
        return list(range(num_pages))
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            pages.update(range(int(start) - 1, int(end)))
        else:
            pages.add(int(part) - 1)
    return sorted(p for p in pages if 0 <= p < num_pages)


def render_page(page, dpi):
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def ocr_images(processor, model, images, prompt, max_new_tokens):
    # Batch the images and prompts for processing
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]

    # Left-padding is required for batched causal-LM generation: it keeps
    # every row's prompt ending at the same column, so a single fixed-offset
    # slice (below) trims the new tokens correctly for the whole batch.
    processor.tokenizer.padding_side = "left"
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)


    # repetition_penalty/no_repeat_ngram_size guard against greedy decoding
    # getting stuck in a repetition loop on visually repetitive content
    # (e.g. table-of-contents dot leaders), running to max_new_tokens instead
    # of stopping normally.
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.2,
    )

    # Number of tokens in the prompt (text + image)
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    texts = [text.strip() for text in processor.batch_decode(trimmed, skip_special_tokens=True)]
    return texts, prompt_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Output text file (default: <pdf>.txt)")
    parser.add_argument("--model", default="zai-org/GLM-OCR",
                         help='HF repo id, e.g. "zai-org/GLM-OCR" (default)')
    parser.add_argument("--prompt", default="Text Recognition:",
                         help='Task prompt. For GLM-OCR use e.g. "Text Recognition:" (default), '
                              '"Table Recognition:", "Formula Recognition:"')
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI for PDF pages")
    parser.add_argument("--pages", default=None,
                         help='Page range, e.g. "1-3,5" (1-indexed). Default: all pages')
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--page-separator", default="\n\n----- page {page} -----\n\n")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    output_path = args.output or args.pdf.with_suffix(".txt")

    doc = pymupdf.open(args.pdf)
    page_indices = parse_page_spec(args.pages, doc.page_count)
    if not page_indices:
        doc.close()
        sys.exit("No valid pages selected")

    device = 0 if torch.cuda.is_available() else -1  #  GPU index or -1 for CPU
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Loading {args.model} on {'cuda:0' if device == 0 else 'cpu'} ({device_name}) ...")
    if device == 0:
        torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, device_map="auto")
    print(f"Model loaded in {time.time() - t0:.1f}s on {model.device}")

    print(f"Rendering {len(page_indices)} page(s) at {args.dpi} DPI ...")
    images = [render_page(doc[page_idx], args.dpi) for page_idx in page_indices]
    doc.close()

    print(f"OCR-ing {len(images)} page(s) as a single batch ...")
    texts, prompt_lengths = ocr_images(processor, model, images, args.prompt, args.max_new_tokens)
    results = [
        args.page_separator.format(page=page_idx + 1) + text
        for page_idx, text in zip(page_indices, texts)
    ]

    output_path.write_text("\n".join(results).strip() + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    # Print token counts and memory usage
    response_tokens = [len(processor.tokenizer.encode(text)) for text in texts]
    total_tokens = sum(prompt_lengths) + sum(response_tokens)
    print(f"Prompt tokens per page (text + image): {prompt_lengths}, "
          f"response tokens per page: {response_tokens}, "
          f"total tokens: {total_tokens}")
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Model Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    print(f"Peak RSS since process start: {peak_kb / 1e6:.2f} GB")
    if device == 0:
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

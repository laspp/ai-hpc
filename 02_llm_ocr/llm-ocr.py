#!/usr/bin/env python3
"""Extract text from a PDF using a vision-language OCR model.

Usage:
    python llm-ocr.py input.pdf
    python llm-ocr.py input.pdf -o output.txt --pages "1-3,5"
    python llm-ocr.py input.pdf --model GaMS-Beta/SVILA-1-4B \\
        --prompt "Natančno prepiši vse besedilo s slike, brez dodatnega komentarja."
    python llm-ocr.py input.pdf --prompt "Table Recognition:" --dpi 300
"""

import argparse
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
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


def render_page(page, dpi, max_long_side):
    zoom = dpi / 72.0
    if max_long_side:
        long_side_pt = max(page.rect.width, page.rect.height)
        zoom = min(zoom, max_long_side / long_side_pt)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def load_model(model_id, dtype, attn_implementation):
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {"device_map": "auto", "dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return processor, model


def ocr_image(processor, model, image, prompt, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


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
    parser.add_argument("--max-long-side", type=int, default=1280,
                         help="Cap the longer side of each rendered page to this many pixels "
                              "(overrides --dpi if it would exceed this). "
                              "Set to 0 to disable and rely on --dpi alone.")
    parser.add_argument("--pages", default=None,
                         help='Page range, e.g. "1-3,5" (1-indexed). Default: all pages')
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default=None,
                         help='e.g. "kernels-community/flash-attn2" if flash-attn is installed')
    parser.add_argument("--page-separator", default="\n\n----- page {page} -----\n\n")
    args = parser.parse_args()

    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    output_path = args.output or args.pdf.with_suffix(".txt")

    doc = fitz.open(args.pdf)
    page_indices = parse_page_spec(args.pages, doc.page_count)
    if not page_indices:
        doc.close()
        sys.exit("No valid pages selected")

    has_gpu = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_gpu else "CPU"
    print(f"Loading {args.model} on {'cuda:0' if has_gpu else 'cpu'} ({device_name}) ...")
    t0 = time.time()
    processor, model = load_model(args.model, args.dtype, args.attn_implementation)
    print(f"Model loaded in {time.time() - t0:.1f}s on {model.device}")

    results = []
    for i, page_idx in enumerate(page_indices, start=1):
        print(f"Rendering and OCR-ing page {page_idx + 1} ({i}/{len(page_indices)}) "
              f"at {args.dpi} DPI (max {args.max_long_side}px long side) ...")
        image = render_page(doc[page_idx], args.dpi, args.max_long_side)
        text = ocr_image(processor, model, image, args.prompt, args.max_new_tokens)
        results.append(args.page_separator.format(page=page_idx + 1) + text)
    doc.close()

    output_path.write_text("\n".join(results).strip() + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

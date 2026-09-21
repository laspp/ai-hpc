#!/usr/bin/env python3
"""
Download a model from the Hugging Face Hub and run a single query against it.

Usage:
    python query_model.py --prompt "What is the capital of Slovenia?"
    python query_model.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --prompt "Explain KV-cache in one sentence." --max-new-tokens 64
"""
import argparse
import time
import resource

import torch
from transformers import GenerationConfig, pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a Hugging Face chat model.")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model repo id (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt", default="What is the capital of Slovenia?",
        help="User prompt to send to the model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Select device: GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1  #  GPU index or -1 for CPU
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print(f"Loading '{args.model}' on {'cuda:0' if device == 0 else 'cpu'} ({device_name}) ...")
    if device == 0:
        torch.cuda.reset_peak_memory_stats(0)
    # Create the pipline
    t0 = time.time()
    pipe = pipeline(
        task="text-generation",
        model=args.model,
        dtype="auto",
        device=device,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s on {pipe.model.device}")
    print(next(pipe.model.parameters()).dtype)
    
    # Construct the query input
    query = [
        {"role": "user", "content": args.prompt},
    ]

    # Configure generation parameters
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_length=None,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    t0 = time.time()

    # Run inference
    outputs = pipe(
        query,
        generation_config=generation_config,
        clean_up_tokenization_spaces=False,
    )
    
    gen_time = time.time() - t0

    # Print the response
    response = outputs[0]["generated_text"][-1]["content"]
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Response ===")
    print(response.strip())
    print(f"\ngenerated in {gen_time:.1f}s ")
    
    # Print token counts and memory usage
    prompt_tokens = len(pipe.tokenizer.encode(args.prompt))
    response_tokens = len(pipe.tokenizer.encode(response))
    print(f"Prompt tokens: {prompt_tokens}, response tokens: {response_tokens}, total tokens: {prompt_tokens + response_tokens}")
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Model Memory footprint: {pipe.model.get_memory_footprint() / 1e9:.2f} GB")
    print(f"Peak RSS since process start: {peak_kb / 1e6:.2f} GB")
    if device == 0:
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
if __name__ == "__main__":
    main()

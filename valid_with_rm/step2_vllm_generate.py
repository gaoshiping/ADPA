import torch
import argparse
import json
import os
from vllm import LLM, SamplingParams
from utils import load_single_dataset

def main():
    parser = argparse.ArgumentParser(description="Batch inference generation using vLLM (Chat Template supported)")
    
    # 1. Model Path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    
    # 2. Generation Hyperparameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--stop", type=str, nargs='*', default=None, help="List of stop sequences")
    
    # 3. Dataset Parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input dataset (jsonl)")
    
    # 4. Key Configuration
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Column name for the prompt in the dataset")
    parser.add_argument("--response_key", type=str, default="response", help="Column name to save the generated response")
    
    # Output Path
    parser.add_argument("--output_path", type=str, default="generation_results.jsonl", help="Path to save output results")
    
    args = parser.parse_args()

    # --- Initialize vLLM Sampling Params ---
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        n=1  # Generate exactly one response per prompt
    )

    # Initialize LLM
    # tensor_parallel_size automatically adapts to the number of available GPUs
    llm = LLM(
        model=args.model_path, 
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count()
    )

    # --- Load Dataset ---
    print(f"Loading dataset from: {args.dataset_path}...")
    dataset = load_single_dataset(args.dataset_path)
    
    # dataset[args.prompt_key] entries are expected to be List[Dict] (role-content format)
    raw_prompts = dataset[args.prompt_key]

    # --- Apply Chat Template ---
    print("Applying Chat Template...")
    # Use the tokenizer integrated within vLLM
    tokenizer = llm.get_tokenizer()
    
    # Convert List[Dict] to rendered prompt strings
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in raw_prompts
    ]

    # --- Execute Inference ---
    print(f"Starting generation (Total: {len(formatted_prompts)} items)...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # --- Extract Results and Save ---
    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            item = dataset[i]
            # Extract the generated text content
            generated_text = output.outputs[0].text
            # Inject the new result into the original dictionary
            item[args.response_key] = generated_text
            
            # Write to jsonl, preserving original format
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Inference complete! Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()


"""
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 \
python ./step2_vllm_generate.py \
    --model_path  /path/to/HuggingFaceH4-zephyr-7b-beta \ # replace with your model path
    --dataset_path ./dpomix7k_validation_prompts.jsonl \
    --prompt_key prompt \
    --response_key response \
    --output_path ./dpomix7k_validation_results.jsonl \
    --temperature 0.0 \
    --max_tokens 512 \
    --stop "</s>"
"""
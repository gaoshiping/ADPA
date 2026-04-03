import argparse
import json
import os
from utils import load_single_dataset

def get_dpomix7k_validation_prompts():
    """
    In the DPO dataset, 'chosen' is a complete conversation list.
    We need to extract the conversation history excluding the final model response.
    """
    dataset_path = "argilla/dpo-mix-7k"
    dataset = load_single_dataset(dataset_path, dataset_split="test")
    
    # Extract the conversation list up to the second to last element [:-1]
    # Format example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    prompts = list(map(lambda chosen: chosen[:-1], dataset["chosen"]))
    return prompts


def get_helpsteer2_validation_prompts():
    """
    The 'prompt' field in the HelpSteer2 dataset contains duplicates.
    We deduplicate the raw text first, then wrap it into the role-content format.
    Deduplication is necessary due to a large number of repeated prompts in HelpSteer2.
    """
    dataset_path = "nvidia/HelpSteer2"
    dataset = load_single_dataset(dataset_path, dataset_split="validation")
    
    # 1. Get the list of raw prompts
    raw_prompts = dataset["prompt"]
    
    # 2. Use a set for deduplication, then convert back to list 
    # Using sorted() ensures consistent order across different runs
    unique_prompts = sorted(list(set(raw_prompts)))
    
    print(f"HelpSteer2: Original data {len(raw_prompts)} items, after deduplication {len(unique_prompts)} items.")

    # 3. Wrap the unique raw text into the role-content list format
    prompts = list(map(
        lambda p: [{"role": "user", "content": p}], 
        unique_prompts
    ))
    
    return prompts


def save_to_jsonl(data, file_path):
    """General function to save data to a JSONL file"""
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            # 'entry' is already in List[Dict] format
            f.write(json.dumps({"prompt": entry}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./", help="The output directory.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Process DPO-Mix-7k
    print("Processing DPO-Mix-7k...")
    dpomix7k_prompts = get_dpomix7k_validation_prompts()
    save_to_jsonl(dpomix7k_prompts, os.path.join(args.output_dir, "dpomix7k_validation_prompts.jsonl"))

    # 2. Process HelpSteer2
    print("Processing HelpSteer2...")
    helpsteer2_prompts = get_helpsteer2_validation_prompts()
    save_to_jsonl(helpsteer2_prompts, os.path.join(args.output_dir, "helpsteer2_validation_prompts.jsonl"))

    print(f"Done! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
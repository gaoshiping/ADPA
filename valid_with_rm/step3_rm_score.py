import argparse
import os.path
import statistics
from statistics import mean
from typing import Dict, List
import json
import time
import torch
from torch import nn
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM, 
    pipeline, 
    AutoModel
)

# Initialize Accelerator for multi-GPU support
accelerator = Accelerator()

# ==================== Reward Model Scoring Classes ====================

class ScoreByCE:
    """Cross-Entropy based scoring (Log-likelihood of the sequence)."""
    def __init__(self, load_from):
        if load_from is None: load_from = "wandb/mistral-7b-zephyr-dpo"
        self.device = accelerator.device
        self.rm_tokenizer = AutoTokenizer.from_pretrained(load_from, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            load_from, device_map=self.device, trust_remote_code=True, torch_dtype=torch.float16)

    def get_logp(self, logits, labels):
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        logits = logits.view(-1, logits.shape[-1])
        labels = labels.view(-1).to(logits.device)
        loss = loss_fct(logits, labels)
        return loss

    def __call__(self, chats: List[List[Dict]]):
        rewards = []
        with torch.no_grad():
            for chat in chats:
                inputs_str = self.rm_tokenizer.apply_chat_template(chat, tokenize=False)
                inputs = self.rm_tokenizer.encode(inputs_str, return_tensors="pt", max_length=2048, truncation=True).to(self.model.device)
                output = self.model(inputs)
                nll_loss = self.get_logp(output.logits, inputs)
                rewards.append(- nll_loss.item())
        return rewards

class ScoreByArmoRM:
    """Scoring using ArmoRM (Llama-3 based)."""
    def __init__(self, load_from):
        if load_from is None: load_from = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        self.device = accelerator.device
        self.rm_tokenizer = AutoTokenizer.from_pretrained(load_from, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_from, device_map=self.device, trust_remote_code=True, torch_dtype=torch.bfloat16)

    def __call__(self, chats: List[List[Dict]]):
        rewards = []
        with torch.no_grad():
            for chat in chats:
                input_ids = self.rm_tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
                output = self.model(input_ids)
                rewards.append(output.score.cpu().float().item())
        return rewards

class ScoreByFsfairx:
    """Scoring using FsfairX RM."""
    def __init__(self, load_from):
        if load_from is None: load_from = "sfairXC/FsfairX-LLaMA3-RM-v0.1"
        self.rm_tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model=load_from,
            device=accelerator.device,
            tokenizer=self.rm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        self.pipe_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    def __call__(self, chats: List[List[Dict]]):
        rewards = []
        for chat in chats:
            test_texts = [self.rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(self.rm_tokenizer.bos_token, "")]
            pipe_outputs = self.rm_pipe(test_texts, **self.pipe_kwargs)
            rewards.append(pipe_outputs[0][0]["score"])
        return rewards

class ScoreByEurus:
    """Scoring using Eurus-RM."""
    def __init__(self, load_from):
        if load_from is None: load_from = "openbmb/Eurus-RM-7b"
        self.tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.model = AutoModel.from_pretrained(load_from, trust_remote_code=True).to(accelerator.device)

    def __call__(self, chats: List[List[Dict]]):
        rewards = []
        with torch.no_grad():
            for chat in chats:
                inputs_str = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                inputs = self.tokenizer(inputs_str, return_tensors="pt").to(accelerator.device)
                reward = self.model(**inputs).item()
                rewards.append(reward)
        return rewards

class ScoreByRmMistral:
    """Scoring using Mistral-7B RM."""
    def __init__(self, load_from):
        if load_from is None: load_from = "weqweasdas/RM-Mistral-7B"
        self.rm_tokenizer = AutoTokenizer.from_pretrained(load_from)
        self.rm_pipe = pipeline(
            "sentiment-analysis",
            model=load_from,
            device=accelerator.device,
            tokenizer=self.rm_tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        self.pipe_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1}

    def __call__(self, chats: List[List[Dict]]):
        rewards = []
        for chat in chats:
            test_texts = [self.rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(self.rm_tokenizer.bos_token, "")]
            pipe_outputs = self.rm_pipe(test_texts, **self.pipe_kwargs)
            rewards.append(pipe_outputs[0][0]["score"])
        return rewards

# ==================== Utility Functions ====================

def batch_prompts(prompts, batch_size=16):
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

def main():
    parser = argparse.ArgumentParser(description="Score model responses using various Reward Models.")
    parser.add_argument('-d', '--data', type=str, required=True, help="Path to input jsonl file.")
    parser.add_argument('-b', "--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument('-p', '--prompt_key', type=str, default="prompt", help="Key for prompt List[Dict].")
    parser.add_argument('-r', '--response_key', type=str, default="response", help="Key for response string.")
    parser.add_argument('-o', '--out', type=str, required=True, help="Path to save output jsonl file.")
    parser.add_argument("-m", "--reward_model", type=str, choices=["fsfairx", "eurus", "rmmistral", "armorm", "ce"], default="armorm", help="Reward model type.")
    parser.add_argument("-l", "--load_from", type=str, default=None, help="Local path or HF path for the RM.")
    parser.add_argument("-s", "--max_sample", type=int, default=None, help="Max samples to process (for debugging).")
    parser.add_argument("-k", "--rm_score_key", type=str, default="rm_score", help="Field name for the resulting score.")
    args = parser.parse_args()

    # 1. Load Dataset
    if accelerator.is_main_process:
        print(f"Loading data from {args.data}...")
    
    dataset = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    
    if args.max_sample:
        dataset = dataset[:args.max_sample]

    # 2. Merge Prompt and Response into a full chat history
    full_chats = []
    for item in dataset:
        # prompt is List[Dict], response is str
        chat = item[args.prompt_key] + [{"role": "assistant", "content": item[args.response_key]}]
        full_chats.append(chat)

    # 3. Initialize chosen Reward Model
    if args.reward_model == "eurus":
        score_by_rm = ScoreByEurus(load_from=args.load_from)
    elif args.reward_model == "rmmistral":
        score_by_rm = ScoreByRmMistral(load_from=args.load_from)
    elif args.reward_model == "armorm":
        score_by_rm = ScoreByArmoRM(load_from=args.load_from)
    elif args.reward_model == "ce":
        score_by_rm = ScoreByCE(load_from=args.load_from)
    else:
        score_by_rm = ScoreByFsfairx(load_from=args.load_from)

    # 4. Multi-GPU Distributed Inference
    accelerator.wait_for_everyone()
    start_time = time.time()

    with accelerator.split_between_processes(full_chats) as local_chats:
        local_results = []
        chat_batches = batch_prompts(local_chats, batch_size=args.batch_size)
        
        for batch in tqdm(chat_batches, disable=not accelerator.is_local_main_process):
            scores = score_by_rm(batch)
            local_results.extend(scores)
        
        # Wrap results in a list for gather_object
        wrapped_results = [{"scores": local_results}]

    # 5. Gather and aggregate results from all processes
    all_gathered_results = gather_object(wrapped_results)

    if accelerator.is_main_process:
        all_scores = []
        for res in all_gathered_results:
            all_scores.extend(res["scores"])

        # Validate alignment
        assert len(all_scores) == len(dataset), "Mismatch between score count and dataset size!"

        # 6. Assign scores and calculate average
        for i in range(len(dataset)):
            dataset[i][args.rm_score_key] = all_scores[i]

        avg_score = mean(all_scores)
        print(f"\n" + "="*30)
        print(f"RM Type: {args.reward_model}")
        print(f"Total Scored: {len(all_scores)}")
        print(f"Average RM Score: {avg_score:.4f}")
        print(f"Time Elapsed: {time.time() - start_time:.2f}s")
        print("="*30)

        # 7. Save results to JSONL
        with open(args.out, "w", encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Results saved to: {args.out}")

if __name__ == '__main__':
    main()


"""
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes 4 ./step3_rm_score.py \
    --data ./dpomix7k_validation_results.jsonl \
    --reward_model fsfairx \
    --load_from /path/to/sfairXC-FsfairX-LLaMA3-RM-v0.1/ \
    --prompt_key prompt \
    --response_key response \
    --rm_score_key fsfairx_score \
    --batch_size 1 \
    --out ./dpomix7k_final_scores.jsonl

"""

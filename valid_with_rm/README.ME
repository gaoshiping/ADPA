# Validation with Reward Model (RM Scoring)

This directory contains the complete pipeline for evaluating and selecting the optimal training checkpoints using a Reward Model (RM). This process corresponds to the checkpoint selection strategy described in **Figure 4** of the paper, where the best-performing model is identified based on RM scores calculated over a validation set.

## Workflow Overview

The evaluation pipeline consists of three independent steps:
1. **Step 1: Data Preparation** - Extract validation prompts from datasets (e.g., `dpo-mix-7k`). We have already provide the prompts in `role-content` format for DPO-MIX-7K and Helpsteer2.
2. **Step 2: Model Inference** - Load your trained checkpoint using vLLM to generate responses for the extracted prompts.
3. **Step 3: RM Scoring** - Use a specified Reward Model (e.g., `FsfairX-LLaMA3-RM-v0.1`) to score the generated responses and calculate the average.

---

## Environment Setup

Before running the code, ensure the following core dependencies are installed:

```bash
pip install vllm accelerate transformers datasets tqdm
```

---

## Quick Start

### Step 1: Prepare Validation Prompts

This script automatically downloads and extracts the test split of `argilla/dpo-mix-7k` (750 items) and the deduplicated validation data from `nvidia/HelpSteer2`, saving them in `jsonl` format.

```bash
python step1_prepare_validation.py --output_dir ./
```
*Upon completion, `dpomix7k_validation_prompts.jsonl` and `helpsteer2_validation_prompts.jsonl` will be generated in the current directory.*

### Step 2: Generate Responses via vLLM

Load your trained checkpoint to perform inference on the validation prompts. Replace `--model_path` with the actual absolute path to your checkpoint.

```bash
VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 \
python step2_vllm_generate.py \
    --model_path /path/to/your/checkpoint \
    --dataset_path ./dpomix7k_validation_prompts.jsonl \
    --prompt_key prompt \
    --response_key response \
    --output_path ./dpomix7k_validation_results.jsonl \
    --temperature 0.0 \
    --max_tokens 512 \
    --stop "</s>" # modify the <eos> token accordingly.
```

### Step 3: Scoring with Reward Model (RM)

Use the distributed scoring script to evaluate the responses generated in Step 2. The example below uses `FsfairX-LLaMA3-RM-v0.1` as mentioned in the paper. Ensure `--load_from` points to your local RM path or the correct Hugging Face repository.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes 4 step3_rm_score.py \
    --data ./dpomix7k_validation_results.jsonl \
    --reward_model fsfairx \
    --load_from /path/to/sfairXC-FsfairX-LLaMA3-RM-v0.1/ \
    --prompt_key prompt \
    --response_key response \
    --rm_score_key fsfairx_score \
    --batch_size 1 \ # we have only implemented "batchsize=1"
    --out ./dpomix7k_final_scores.jsonl
```

---

## Results Analysis & Model Selection

After the script finishes, the terminal will display statistical information similar to the following:

```text
==============================
RM Type: fsfairx
Total Scored: 750
Average RM Score: xxxx
Time Elapsed: xx.xxs
==============================
```

**Selection Criteria:** You should repeat **Step 2** and **Step 3** for different checkpoints saved during training (e.g., every 0.5 epochs). Record the `Average RM Score` for each. Finally, select the checkpoint with the **highest average score** as the final model for downstream evaluation on MT-Bench or AlpacaEval.

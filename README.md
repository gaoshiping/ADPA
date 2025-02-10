# ADPA Training Pipeline

This repository provides a step-by-step guide for training Small Language Models (SLMs) using **Advantage-Guided Distillation for Preference Alignment (ADPA)**. The process combines two phases:

1. **Dual-Constrained Knowledge Distillation (DCKD)**: Transfers preference knowledge from a teacher model to an SLM.
2. **Advantage-Guided Distillation for Preference Alignment (ADPA)**: Uses a teacher model’s advantage function to further align the student model with human preferences.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Training Procedure](#training-procedure)
   1. [Train the Teacher Model](#1-train-the-teacher-model)
   2. [Supervised Fine-Tuning (SFT) for Student Initialization](#2-supervised-fine-tuning-sft-for-student-initialization)
   3. [DCKD: Dual-Constrained Knowledge Distillation](#3-dckd-dual-constrained-knowledge-distillation)
   4. [ADPA (and ADPA+)](#4-adpa-and-adpa)
4. [References](#references)

---

## Introduction
ADPA is designed to address the limitations of Small Language Models in preference alignment. By learning from a Large Language Model (LLM) teacher, SLMs can better align with human preferences through:
- **DCKD**: Directly transferring knowledge about both “preferred” and “dispreferred” responses from the teacher.
- **ADPA**: Providing token-level advantage signals to the student model so it can better distinguish between desirable and undesirable actions.

If desired, you can combine both approaches in an **ADPA+** pipeline, which first applies DCKD and then refines alignment through the advantage function.

---

## Environment Setup

Below is an example of creating and activating a Conda environment for training:

```bash
# Create and activate a new environment
conda create -n handbook python=3.11
conda activate handbook

# Install PyTorch (CUDA 12.1) and dependencies
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install additional packages needed
pip install trl==0.12.0 peft==0.13.0 huggingface-hub==0.24.7 vllm deepspeed
```

Feel free to adjust versions or CUDA/CPU settings based on your hardware.

---

## Training Procedure

### 1. Train the Teacher Model

We assume you have:
- **An initial teacher model** (e.g., `LLaMA`, `Mistral`, etc.).
- **A dataset** for supervised fine-tuning (SFT).
- **A preference dataset** for alignment (e.g., DPO-style pairs).
- ADPA folder and placed it at `~/ADPA`

Two stages are needed to get a preference-aligned teacher:

1. **SFT Training**: Fine-tune the teacher on your instruction or SFT data.
2. **DPO Training**: Fine-tune the SFT-ed teacher on preference pairs, producing a DPO teacher that better captures human preferences.

Example commands (replace YAML files with your own configurations):

```bash
# (a) SFT training to get the REF teacher
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_sft.yaml

# (b) DPO training on the REF teacher to get the DPO teacher
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/teacher_dpo.yaml
```

### 2. Supervised Fine-Tuning (SFT) for Student Initialization

Next, initialize a **student model** (the smaller model you want to align) with standard supervised fine-tuning. For instance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_sft.py \
  recipes/llama3.2-1b-deita-dpomix/student_sft_init.yaml
```

This gives you a “reference student” from which you can proceed with DCKD and/or ADPA.

### 3. DCKD: Dual-Constrained Knowledge Distillation

**DCKD** aligns the student model by enforcing KL constraints on both the “chosen” (preferred) and “rejected” (dispreferred) responses from the teacher. It helps the student understand what *not* to generate, as well as what *is* correct.

#### 3.1 Extract Teacher Log-Probabilities

First, we precompute the teacher’s log-probabilities on both chosen and rejected responses:

```bash
# Precompute teacher's logits on "chosen" responses for the training set
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
  --num_processes=8 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split train \
  --model data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key chosen \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-train \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-train/results_rank_*.jsonl

# Repeat similarly for:
#   - "chosen" / "rejected" + train/test
# train-rejected
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
  --num_processes=8 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split train \
  --model data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key rejected \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train/results_rank_*.jsonl

# test-chosen
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
  --num_processes=8 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split test \
  --model data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key chosen \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test/results_rank_*.jsonl

# test-rejected
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
  --num_processes=8 \
  --main_process_port 29501 \
  utils/precompute_logits.py \
  --data argilla/dpo-mix-7k \
  --split test \
  --model data/llama3.2-1b-deita-dpomix/dpo_teacher \
  --conversation-key rejected \
  --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
  --user-end "<|eot_id|>" \
  --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
  --assistant-end "<|eot_id|>" \
  --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test \
  --pad-token-id 128001 \
  --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test/results_rank_*.jsonl

```

Then, merge them into a DCKD dataset:

```bash
python utils/merge_logits_dckd_dataset.py \
    --input-dataset-dict          argilla/dpo-mix-7k \
    --teacher-chosen-logp-train   data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-train \
    --teacher-rejected-logp-train data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-train \
    --teacher-chosen-logp-test    data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-chosen-logp-test \
    --teacher-rejected-logp-test  data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-rejected-logp-test \
    --save-to                     data/llama3.2-1b-deita-dpomix/dpomix7k-dckd
```

#### 3.2 Run DCKD

Now, use the merged dataset to train the student with DCKD:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/student_dckd.yaml
```

### 4. ADPA (and ADPA+)

**ADPA** uses an advantage function (the log-prob differences between the DPO teacher and the reference teacher) to give fine-grained, distribution-level signals. You can optionally initialize with the DCKD student (forming **ADPA+**).

#### 4.1 Generate Student Responses

Use the SFT or DCKD-initialized student to produce “rejected” responses (since the “chosen” ones are from the reference ground truth):

```bash
CUDA_VISIBLE_DEVICES=1 \
python utils/vllm_generate.py \
    --model ~/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/student_dckd \
    --data argilla/dpo-mix-7k \
    --dataset_split train \
    --prompt_key chosen \
    --out_dir ~/ADPA/data/llama3.2-1b-deita-dpomix/student_init_self_generation
    --model ~/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/student_dckd \
    --apply_template
    
CUDA_VISIBLE_DEVICES=1 \
python utils/vllm_generate.py \
    --model ~/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/student_dckd \
    --data argilla/dpo-mix-7k \
    --dataset_split test \
    --prompt_key chosen \
    --out_dir ~/ADPA/data/llama3.2-1b-deita-dpomix/student_init_self_generation
    --model ~/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/student_dckd \
    --apply_template
```

Combine these student responses with the original “chosen” references:

```bash
python utils/form_preference_dataset.py \
    --original-dataset  argilla/dpo-mix-7k \
    --rejected-train    data/llama3.2-1b-deita-dpomix/student_init_self_generation/argilla-dpo-mix-7k-train.jsonl \
    --rejected-test     data/llama3.2-1b-deita-dpomix/student_init_self_generation/argilla-dpo-mix-7k-test.jsonl \
    --output-dir        data/llama3.2-1b-deita-dpomix/student_adpa_dataset_original
```

#### 4.2 Compute Advantage (Teacher vs. Reference)

Get the log-probabilities of:
- **DPO teacher** (π<sub>dpo</sub>) on the student’s “rejected” responses
- **Reference teacher** (π<sub>ref</sub>) on the same responses

```bash
# dpoteacher-train-student
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 29501 \
    utils/precompute_logits.py \
    --data data/llama3.2-1b-deita-dpomix/student_adpa_dataset_original \
    --split train \
    --model data/llama3.2-1b-deita-dpomix/dpo_teacher \
    --conversation-key rejected \
    --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    --user-end "<|eot_id|>" \
    --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
    --assistant-end "<|eot_id|>" \
    --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-train-student \
    --pad-token-id 128001 \
    --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-train-student/results_rank_*.jsonl

# dpoteacher-test-student
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 29501 \
    utils/precompute_logits.py \
    --data data/llama3.2-1b-deita-dpomix/student_adpa_dataset_original \
    --split test \
    --model data/llama3.2-1b-deita-dpomix/dpo_teacher \
    --conversation-key rejected \
    --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    --user-end "<|eot_id|>" \
    --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
    --assistant-end "<|eot_id|>" \
    --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-test-student \
    --pad-token-id 128001 \
    --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-test-student/results_rank_*.jsonl

# refteacher-train-student
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 29501 \
    utils/precompute_logits.py \
    --data data/llama3.2-1b-deita-dpomix/student_adpa_dataset_original \
    --split train \
    --model ~/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/student_dckd \
    --conversation-key rejected \
    --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    --user-end "<|eot_id|>" \
    --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
    --assistant-end "<|eot_id|>" \
    --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-refteacher-train-student \
    --pad-token-id 128001 \
    --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-refteacher-train-student/results_rank_*.jsonl

# refteacher-test-student
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port 29501 \
    utils/precompute_logits.py \
    --data data/llama3.2-1b-deita-dpomix/student_adpa_dataset_original \
    --split test \
    --model ~/ADPA-OpenSource/data/llama3.2-1b-deita-dpomix/student_dckd \
    --conversation-key rejected \
    --user-begin "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
    --user-end "<|eot_id|>" \
    --assistant-begin "<|start_header_id|>assistant<|end_header_id|>\n\n" \
    --assistant-end "<|eot_id|>" \
    --save-to data/llama3.2-1b-deita-dpomix/dpomix7k-refteacher-test-student \
    --pad-token-id 128001 \
    --max-tokens-per-batch 2048
rm data/llama3.2-1b-deita-dpomix/dpomix7k-refteacher-test-student/results_rank_*.jsonl
```

Then merge them:

```bash
python utils/merge_logits_adpa_dataset.py \
    --input-dataset-dict argilla/dpo-mix-7k \
    --dpo-teacher-logp-train ~/ADPA/data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-train-student \
    --ref-teacher-logp-train ~/ADPA/data/llama3.2-1b-deita-dpomix/dpomix7k-refteacher-train-student \
    --dpo-teacher-logp-test  ~/ADPA/data/llama3.2-1b-deita-dpomix/dpomix7k-dpoteacher-test-student \
    --ref-teacher-logp-test  ~/ADPA/data/llama3.2-1b-deita-dpomix/dpomix7k-refteacher-test-student \
    --save-to ~/ADPA/data/llama3.2-1b-deita-dpomix/adpa_dataset \
    --logits-key rejected_compressed_probs \
    --label-key rejected_labels \
    --output-key rejected_margin_logp_every
```

#### 4.3 Run ADPA

Finally, run ADPA training:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
ACCELERATE_LOG_LEVEL=info \
DS_SKIP_CUDA_CHECK=1 \
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_config/deepspeed_zero3.yaml \
  scripts/run_distill_dpo.py \
  recipes/llama3.2-1b-deita-dpomix/student_adpa.yaml
```

If your student was already trained by DCKD, this step becomes **ADPA+**, providing potentially stronger alignment.

---

## References
Please refer to the paper and its references.
```bibtex
@inproceedings{
    gao2025advantageguided,
    title={Advantage-Guided Distillation for Preference Alignment in Small Language Models},
    author={Shiping Gao, Fanqi Wan, Jiajian Guo, Xiaojun Quan, Qifan Wang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025}
}
```
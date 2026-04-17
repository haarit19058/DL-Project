#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Start Script - Minimal Working Example
=============================================

This script demonstrates the core concepts with minimal samples
for quick testing (< 30 minutes on a single GPU)

Use this to:
1. Verify your setup works
2. Understand the pipeline flow
3. Test before running the full pipeline
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from transformers import TrainingArguments
import re
import os
import tempfile
import subprocess
from itertools import islice

print("="*70)
print(" QUICK START - MINIMAL WORKING EXAMPLE")
print("="*70)

# ==========================================
# CONFIGURATION (Small scale for testing)
# ==========================================
MAX_SEQ_LENGTH = 2048
SFT_SAMPLES = 100  # Very small for quick testing
GRPO_SAMPLES = 50
MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

print(f"\n📝 Configuration:")
print(f"   SFT Samples: {SFT_SAMPLES}")
print(f"   GRPO Samples: {GRPO_SAMPLES}")
print(f"   Model: {MODEL_NAME}")

# ==========================================
# STEP 1: LOAD MODEL
# ==========================================
print("\n" + "="*70)
print("STEP 1: Loading Model")
print("="*70)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
)

print("✓ Model loaded with LoRA")

# ==========================================
# STEP 2: PREPARE SFT DATASET
# ==========================================
print("\n" + "="*70)
print("STEP 2: Preparing SFT Dataset")
print("="*70)

# Load minimal dataset
ds_stream = load_dataset(
    "open-r1/codeforces-cots",
    name="solutions_py_decontaminated",
    split="train",
    streaming=True
)

# Filter for 'A' problems (easiest)
filtered = ds_stream.filter(lambda x: x['id'].endswith('A'))
data_list = list(islice(filtered, SFT_SAMPLES + 20))

# Format for SFT
def format_sft(item):
    messages = item.get('messages', [])
    
    # Simple text formatting
    text = ""
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    
    return {'text': text, 'id': item.get('id', '')}

formatted_data = [format_sft(item) for item in data_list]
dataset = Dataset.from_list(formatted_data)
split = dataset.train_test_split(test_size=0.2, seed=3407)

train_dataset = split['train']
eval_dataset = split['test']

print(f"✓ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ==========================================
# STEP 3: SFT TRAINING
# ==========================================
print("\n" + "="*70)
print("STEP 3: SFT Training (Quick)")
print("="*70)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,  # Just 1 epoch for quick test
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        output_dir="outputs_quickstart/sft",
        save_strategy="no",  # Don't save for quick test
        eval_strategy="no",
        report_to="none",
    ),
)

print("🚀 Starting SFT training...")
trainer.train()
print("✓ SFT complete!")

# ==========================================
# STEP 4: PREPARE GRPO DATASET
# ==========================================
print("\n" + "="*70)
print("STEP 4: Preparing GRPO Dataset")
print("="*70)

# Load smaller dataset for GRPO
grpo_data = load_dataset(
    "open-r1/codeforces-cots",
    split=f"train[:{GRPO_SAMPLES}]"
)
grpo_data = grpo_data.filter(lambda x: x['id'].endswith('A'))

def format_grpo(row):
    examples = row.get('examples', [])
    input_tests = [str(ex['input']) for ex in examples]
    output_tests = [str(ex['output']) for ex in examples]
    
    prompt = (
        "Think step-by-step in <think></think> tags, then provide code.\n\n"
        f"{row['prompt']}\n\n<think>\n"
    )
    
    return {
        "prompt": prompt,
        "input_tests": input_tests,
        "output_tests": output_tests,
    }

grpo_dataset = grpo_data.map(format_grpo, num_proc=4)
print(f"✓ GRPO dataset: {len(grpo_dataset)} samples")

# ==========================================
# STEP 5: DEFINE SIMPLE REWARDS
# ==========================================
print("\n" + "="*70)
print("STEP 5: Setting up Reward Functions")
print("="*70)

def format_reward(completions, **kwargs):
    """Check for <think> tags and code block"""
    rewards = []
    for c in completions:
        content = c[0]['content'] if isinstance(c, list) else c
        
        has_think = bool(re.search(r"</think>", content))
        has_code = bool(re.search(r"```python\n.*?```", content, re.DOTALL))
        
        rewards.append(0.5 if (has_think and has_code) else 0.0)
    return rewards

def correctness_reward(completions, input_tests, output_tests, **kwargs):
    """Execute code and check correctness"""
    rewards = []
    
    for comp, inputs, outputs in zip(completions, input_tests, output_tests):
        content = comp[0]['content'] if isinstance(comp, list) else comp
        
        # Extract code
        match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if not match:
            rewards.append(-0.5)
            continue
        
        code = match.group(1).strip()
        
        # Test code
        score = 0
        total = len(inputs)
        
        for test_in, test_out in zip(inputs, outputs):
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    tmp_path = f.name
                
                result = subprocess.run(
                    ['python3', tmp_path],
                    input=test_in,
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                
                if result.stdout.strip() == str(test_out).strip():
                    score += 1
                
                os.remove(tmp_path)
            except:
                pass
        
        # Reward based on pass rate
        if score == total and total > 0:
            rewards.append(2.0)
        elif score > total / 2:
            rewards.append(0.5 + (score / total))
        else:
            rewards.append(-0.5)
    
    return rewards

print("✓ Reward functions ready")

# ==========================================
# STEP 6: GRPO TRAINING
# ==========================================
print("\n" + "="*70)
print("STEP 6: GRPO Training (Quick)")
print("="*70)

grpo_config = GRPOConfig(
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=2,  # Reduced for speed
    max_completion_length=1024,
    num_train_epochs=1,  # Just 1 epoch
    logging_steps=5,
    save_strategy="no",
    output_dir="outputs_quickstart/grpo",
    report_to="none",
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    processing_class=tokenizer,
    reward_funcs=[format_reward, correctness_reward],
    train_dataset=grpo_dataset,
)

print("🚀 Starting GRPO training...")
grpo_trainer.train()
print("✓ GRPO complete!")

# ==========================================
# STEP 7: TEST THE MODEL
# ==========================================
print("\n" + "="*70)
print("STEP 7: Testing Trained Model")
print("="*70)

# Test on a sample problem
test_sample = grpo_dataset[0]
prompt = test_sample['prompt']

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id,
)

generated = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)

print("\n--- Generated Solution ---")
print(generated[:500] + "...\n")

# Evaluate
full_completion = "<think>\n" + generated
fmt_score = format_reward([full_completion])[0]
corr_score = correctness_reward(
    [full_completion],
    [test_sample["input_tests"]],
    [test_sample["output_tests"]]
)[0]

print("--- Scores ---")
print(f"Format:      {fmt_score:.2f}")
print(f"Correctness: {corr_score:.2f}")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*70)
print("🎉 QUICK START COMPLETE!")
print("="*70)
print("\nNext Steps:")
print("1. Review the generated solution above")
print("2. If it looks good, run the full pipeline: python qwen3_cot_pipeline.py")
print("3. Adjust hyperparameters in PipelineConfig if needed")
print("4. Read GUIDE.md for detailed documentation")
print("\n" + "="*70)

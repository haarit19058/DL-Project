#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen 3 4B End-to-End Training Pipeline with CoT
================================================
Step 1: SFT on CodeForces CoT dataset
Step 2: GRPO reinforcement learning
Step 3: Benchmarking & Evaluation
Step 4: Inference-time training (context distillation)

Note: Optimized for lightweight system memory footprint (Arch Linux friendly)
and scaled for continuous Tiny CM model reasoning improvements.
"""

import os
import re
import sys
import json
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from itertools import islice
import numpy as np

# ==========================================
# INSTALLATION & SETUP
# ==========================================
def install_dependencies():
    packages = [
        "unsloth",
        "trl==0.24.0",
        "datasets",
        "transformers==5.5.0",
        "bitsandbytes",
        "accelerate",
        "vllm",
        "huggingface_hub",
        "mergekit",
        "flashinfer-cubin==0.6.4",
        "wandb",
        "nltk",
    ]
    for package in packages:
        os.system(f"pip install {package}")

# Uncomment to install dependencies
# install_dependencies()

# ==========================================
# IMPORTS
# ==========================================
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
from transformers import TrainingArguments
from transformers.utils import logging

logging.set_verbosity_info()

# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class PipelineConfig:
    """Centralized configuration for the entire pipeline"""
    
    model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    max_seq_length: int = 4096  
    load_in_4bit: bool = True
    
    # LoRA configuration
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # SFT configuration (Streaming requires max_steps instead of epochs)
    sft_max_steps: int = 5000       # Adjust based on how long you want to train
    sft_eval_samples: int = 500
    sft_batch_size: int = 2
    sft_grad_accum: int = 8         # Effective batch = 16
    sft_learning_rate: float = 2e-4
    sft_warmup_ratio: float = 0.1
    
    # GRPO configuration (Streaming requires max_steps instead of epochs)
    grpo_max_steps: int = 2000      # Adjust based on RL convergence
    grpo_batch_size: int = 2
    grpo_grad_accum: int = 8
    grpo_learning_rate: float = 5e-6
    grpo_num_generations: int = 4
    grpo_max_completion_length: int = 2048

    # Dataset configuration
    dataset_name: str = "open-r1/codeforces-cots"
    dataset_subset: str = "solutions_decontaminated"
    problem_filter: Tuple[str, ...] = ("A", "B", "C")  # Training on Expert/Div2 level
    
    # Paths
    sft_output_dir: str = "outputs/sft_cot"
    grpo_output_dir: str = "outputs/grpo_cot"
    final_model_dir: str = "outputs/final_model"
    benchmark_dir: str = "outputs/benchmarks"
    
    seed: int = 3407
    report_to: str = "none"

config = PipelineConfig()

# ==========================================
# STEP 1: SFT WITH COT DATASET
# ==========================================

class SFTCoTPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print("="*50)
        print("STEP 1: Loading Model for SFT")
        print("="*50)
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            fast_inference=False,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=True,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        print(f"✓ Model loaded: {self.config.model_name}")
        
    def prepare_sft_dataset(self):
        print("\n" + "="*50)
        print("Preparing Full SFT Dataset (Pure Streaming Mode)")
        print("="*50)
        
        # Stream the dataset to keep System RAM at ~0GB
        dataset = load_dataset(
            self.config.dataset_name,
            name=self.config.dataset_subset,
            split="train",
            streaming=True
        )
        
        filtered_ds = dataset.filter(
            lambda x: x['id'].endswith(self.config.problem_filter)
        )
        
        def map_format(item):
            formatted = self._format_cot_example(item)
            if formatted is None:
                return {"text": "", "id": item.get("id", "unknown"), "valid": False}
            return {"text": formatted["text"], "id": formatted["id"], "valid": True}

        mapped_ds = filtered_ds.map(map_format)
        final_ds = mapped_ds.filter(lambda x: x["valid"])
        
        # Split the stream manually for IterableDatasets
        eval_dataset = final_ds.take(self.config.sft_eval_samples)
        train_dataset = final_ds.skip(self.config.sft_eval_samples)
        
        print(f"✓ Streaming pipeline established. Ready to stream full dataset.")
        return train_dataset, eval_dataset
    
    def _format_cot_example(self, item: Dict) -> Optional[Dict]:
        try:
            messages = item.get('messages', [])
            if len(messages) < 2: return None
            
            system_msg, user_msg, assistant_msg = "", "", ""
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'system': system_msg = content
                elif role == 'user': user_msg = content
                elif role == 'assistant': assistant_msg = content
            
            formatted_text = f"""<|im_start|>system\n{system_msg if system_msg else "You are an expert competitive programmer. Think step-by-step and explain your reasoning before providing the solution."}\n<|im_end|>\n<|im_start|>user\n{user_msg}\n\nPlease think through this problem step-by-step inside <think></think> tags, then provide your solution in a code block.\n<|im_end|>\n<|im_start|>assistant\n{assistant_msg}\n<|im_end|>"""
            
            return {'text': formatted_text, 'id': item.get('id', 'unknown')}
        except Exception:
            return None
    
    def train_sft(self, train_dataset, eval_dataset):
        print("\n" + "="*50)
        print("Starting SFT Training")
        print("="*50)
        
        training_args = TrainingArguments(
            per_device_train_batch_size=self.config.sft_batch_size,
            per_device_eval_batch_size=self.config.sft_batch_size,
            gradient_accumulation_steps=self.config.sft_grad_accum,
            max_steps=self.config.sft_max_steps, # Replaced num_train_epochs for streaming
            learning_rate=self.config.sft_learning_rate,
            warmup_ratio=self.config.sft_warmup_ratio,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            max_grad_norm=1.0,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=25,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            output_dir=self.config.sft_output_dir,
            report_to=self.config.report_to,
            seed=self.config.seed,
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,
            args=training_args,
        )
        
        print("\n🚀 Starting SFT training over the full dataset stream...")
        trainer_stats = trainer.train()
        
        print("\n💾 Saving SFT model...")
        self.model.save_pretrained(self.config.sft_output_dir + "/final")
        self.tokenizer.save_pretrained(self.config.sft_output_dir + "/final")
        return trainer_stats

# ==========================================
# STEP 2: GRPO TRAINING WITH ENHANCED REWARDS
# ==========================================

class EnhancedRewardFunctions:
    @staticmethod
    def format_reward(completions, **kwargs) -> List[float]:
        """
        Reward for proper format: <think>...</think> + code block
        
        Returns: 0.0 to 1.0
        """
        rewards = []
        for comp in completions:
            content = comp[0]['content'] if isinstance(comp, list) else comp
            
            score = 0.0
            
            # Check for think tags (0.3 points)
            has_think_open = bool(re.search(r"<think>", content))
            has_think_close = bool(re.search(r"</think>", content))
            if has_think_open and has_think_close:
                score += 0.3
            elif has_think_open or has_think_close:
                score += 0.1  # Partial credit
            
            # Check for code block (0.3 points)
            has_code = bool(re.search(r"```(python|cpp)\n.*?```", content, re.DOTALL))
            if has_code:
                score += 0.3
            
            # Check for proper ordering: think before code (0.2 points)
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            code_match = re.search(r"```(python|cpp)\n(.*?)```", content, re.DOTALL)
            if think_match and code_match:
                if content.index(think_match.group(0)) < content.index(code_match.group(0)):
                    score += 0.2
            
            # Check for complete response (0.2 points)
            if len(content.strip()) > 100:  # Minimum reasonable length
                score += 0.2
            
            rewards.append(score)
        
        return rewards
    
    @staticmethod
    def reasoning_quality_reward(completions, **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            content = comp[0]['content'] if isinstance(comp, list) else comp
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if not think_match:
                rewards.append(-0.2)
                continue
            
            thinking = think_match.group(1).strip()
            score = 0.0
            
            word_count = len(thinking.split())
            if word_count < 20: score += 0.0
            elif word_count < 50: score += 0.1
            elif word_count < 150: score += 0.2 + 0.1 * min((word_count - 50) / 100, 1.0)
            elif word_count < 300: score += 0.3
            else: score += 0.2
            
            logical_indicators = [
                r'\b(first|second|third|then|next|finally)\b', r'\b(because|since|therefore|thus|hence)\b',
                r'\b(if|when|while|for each)\b', r'\b(consider|note|observe|notice)\b',
            ]
            logic_score = sum(min(len(re.findall(p, thinking, re.IGNORECASE)) * 0.05, 0.075) for p in logical_indicators)
            score += min(logic_score, 0.3)
            
            algo_indicators = [r'\b(time complexity|space complexity|O\(.*?\))\b', r'\b(algorithm|approach|strategy|method)\b',
                               r'\b(edge case|corner case|boundary)\b', r'\b(optimize|efficient|improve)\b']
            algo_score = sum(0.05 for p in algo_indicators if re.search(p, thinking, re.IGNORECASE))
            score += min(algo_score, 0.2)
            
            if bool(re.search(r'^\s*[-*•]\s', thinking, re.MULTILINE)) or bool(re.search(r'^\s*\d+[\.)]\s', thinking, re.MULTILINE)):
                score += 0.15
            elif len(thinking.split('\n\n')) > 1:
                score += 0.1
            
            rewards.append(min(score, 1.0))
        return rewards
    
    @staticmethod
    def extract_code(completion: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.search(r"```(python|cpp)\n(.*?)```", completion, re.DOTALL)
        return (match.group(1), match.group(2).strip()) if match else (None, None)
    
    @staticmethod
    def code_quality_reward(completions, **kwargs) -> List[float]:
        rewards = []
        for comp in completions:
            content = comp[0]['content'] if isinstance(comp, list) else comp
            lang, code = EnhancedRewardFunctions.extract_code(content)
            if not code or lang != "python":
                rewards.append(0.0)
                continue
            
            score = 0.0
            if re.search(r'\bdef\s+\w+\s*\(', code): score += 0.1
            if bool(re.search(r'#.*$', code, re.MULTILINE)) or bool(re.search(r'""".*?"""', code, re.DOTALL)): score += 0.1
            if len(set(re.findall(r'\b([a-z_][a-z0-9_]{2,})\b', code))) >= 3: score += 0.1
            if re.search(r'\[.*for.*in.*\]|\(.*for.*in.*\)', code): score += 0.05
            
            lines = [l for l in code.split('\n') if l.strip()]
            if 5 <= len(lines) <= 100: score += 0.1
            
            rewards.append(min(score, 0.5))
        return rewards
    
    @staticmethod
    def correctness_reward(completions, input_tests, output_tests, **kwargs) -> List[float]:
        rewards = []
        for comp, inputs, outputs in zip(completions, input_tests, output_tests):
            content = comp[0]['content'] if isinstance(comp, list) else comp
            lang, code = EnhancedRewardFunctions.extract_code(content)
            
            if not code or not lang:
                rewards.append(-1.0)
                continue
            
            score, total, compilation_failed = 0, len(inputs), False
            
            for test_input, expected_output in zip(inputs, outputs):
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py' if lang == 'python' else '.cpp', delete=False) as tmp:
                        tmp.write(code)
                        tmp_path = tmp.name
                    
                    try:
                        if lang == 'python':
                            result = subprocess.run(['python3', tmp_path], input=test_input, capture_output=True, text=True, timeout=5, check=False)
                        else:
                            exe_path = tmp_path + '.out'
                            compile_result = subprocess.run(['g++', '-std=c++17', tmp_path, '-o', exe_path], capture_output=True, text=True, timeout=10, check=False)
                            if compile_result.returncode != 0:
                                compilation_failed = True
                                break
                            result = subprocess.run([exe_path], input=test_input, capture_output=True, text=True, timeout=5, check=False)
                            os.remove(exe_path)
                        
                        # Fix: Cap output at 10,000 characters to prevent infinite loop RAM leakage
                        actual_output = result.stdout[:10000].strip() if result.stdout else ""
                        if actual_output == str(expected_output).strip():
                            score += 1
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                except subprocess.TimeoutExpired: pass
                except Exception: pass
            
            if compilation_failed: rewards.append(-2.0)
            elif score == total and total > 0: rewards.append(3.0)
            elif score > total / 2: rewards.append(0.5 + 2.0 * (score / total))
            else: rewards.append(-0.5)
        
        return rewards

class GRPOPipeline:
    def __init__(self, config: PipelineConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.rewards = EnhancedRewardFunctions()
    
    def prepare_grpo_dataset(self):
        print("\n" + "="*50)
        print("STEP 2: Preparing GRPO Dataset (Pure Streaming Mode)")
        print("="*50)
        
        dataset = load_dataset(
            self.config.dataset_name,
            name=self.config.dataset_subset,
            split="train",
            streaming=True
        )
        
        filtered = dataset.filter(lambda x: x['id'].endswith(self.config.problem_filter))
        processed_stream = filtered.map(self._format_grpo_example)
        
        return processed_stream

    def _format_grpo_example(self, row: Dict) -> Dict:
        examples = row.get('examples', [])
        prompt = f"You are a competitive programming expert. Think step-by-step inside <think></think> tags, then provide your solution in a code block.\n\n{row['prompt']}\n\n<think>\n"
        return {
            "prompt": prompt,
            "input_tests": [str(ex['input']) for ex in examples],
            "output_tests": [str(ex['output']) for ex in examples],
            "id": row.get('id', 'unknown')
        }
    
    def train_grpo(self, train_dataset):
        print("\n" + "="*50)
        print("Starting GRPO Training")
        print("="*50)
        
        training_args = GRPOConfig(
            learning_rate=self.config.grpo_learning_rate,
            per_device_train_batch_size=self.config.grpo_batch_size,
            gradient_accumulation_steps=self.config.grpo_grad_accum,
            max_steps=self.config.grpo_max_steps, # Replaced num_train_epochs for streaming
            num_generations=self.config.grpo_num_generations,
            max_completion_length=self.config.grpo_max_completion_length,
            loss_type="grpo",
            importance_sampling_level="sequence",
            lr_scheduler_type="cosine",
            weight_decay=0.1,
            max_grad_norm=1.0,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            output_dir=self.config.grpo_output_dir,
            report_to=self.config.report_to,
        )
        
        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.rewards.format_reward,
                self.rewards.reasoning_quality_reward,
                self.rewards.code_quality_reward,
                self.rewards.correctness_reward,
            ],
            train_dataset=train_dataset,
        )
        
        print("\n🚀 Starting GRPO training over the full dataset stream...")
        trainer.train()
        
        print("\n💾 Saving GRPO model...")
        trainer.save_model(self.config.grpo_output_dir + "/final")
        self.tokenizer.save_pretrained(self.config.grpo_output_dir + "/final")

# ==========================================
# STEP 3: BENCHMARKING & EVALUATION
# ==========================================

class BenchmarkPipeline:
    def __init__(self, model, tokenizer, config: PipelineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.rewards = EnhancedRewardFunctions()
    
    def run_benchmark(self, test_dataset, num_samples: int = 100):
        print("\n" + "="*50)
        print("STEP 3: Running Benchmark Evaluation")
        print("="*50)
        
        # Enable Unsloth Native Inference mode to drastically cut VRAM usage and speed up generation
        FastLanguageModel.for_inference(self.model)
        
        results = { 'format_scores': [], 'reasoning_scores': [], 'code_quality_scores': [], 'correctness_scores': [], 'all_tests_passed': 0, 'compilation_failures': 0, 'total_evaluated': 0 }
        
        for i, sample in enumerate(test_dataset):
            if i >= num_samples: break
            
            completion = self._generate_solution(f"Think step-by-step inside <think></think> tags, then provide your solution.\n\n{sample.get('prompt', sample.get('text', ''))}\n\n<think>\n")
            
            results['format_scores'].append(self.rewards.format_reward([completion])[0])
            results['reasoning_scores'].append(self.rewards.reasoning_quality_reward([completion])[0])
            results['code_quality_scores'].append(self.rewards.code_quality_reward([completion])[0])
            c_score = self.rewards.correctness_reward([completion], [sample['input_tests']], [sample['output_tests']])[0]
            results['correctness_scores'].append(c_score)
            
            if c_score >= 2.5: results['all_tests_passed'] += 1
            elif c_score == -2.0: results['compilation_failures'] += 1
            results['total_evaluated'] += 1
            
            if (i + 1) % 10 == 0: print(f"Evaluated {i + 1}/{num_samples} samples...")
        
        stats = self._compute_statistics(results)
        self._save_benchmark_results(stats)
        self._print_benchmark_summary(stats)
        return stats
    
    def _generate_solution(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=2048, pad_token_id=self.tokenizer.pad_token_id, temperature=0.7, top_p=0.9, do_sample=True)
        return "<think>\n" + self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    def _compute_statistics(self, results: Dict) -> Dict:
        return {
            'format': {'mean': np.mean(results['format_scores']), 'std': np.std(results['format_scores'])},
            'reasoning': {'mean': np.mean(results['reasoning_scores']), 'std': np.std(results['reasoning_scores'])},
            'code_quality': {'mean': np.mean(results['code_quality_scores']), 'std': np.std(results['code_quality_scores'])},
            'correctness': {'mean': np.mean(results['correctness_scores']), 'std': np.std(results['correctness_scores'])},
            'pass_rate': results['all_tests_passed'] / results['total_evaluated'] if results['total_evaluated'] > 0 else 0,
            'compilation_fail_rate': results['compilation_failures'] / results['total_evaluated'] if results['total_evaluated'] > 0 else 0,
            'total_evaluated': results['total_evaluated'],
        }
    
    def _save_benchmark_results(self, stats: Dict):
        os.makedirs(self.config.benchmark_dir, exist_ok=True)
        with open(os.path.join(self.config.benchmark_dir, "benchmark_results.json"), 'w') as f: json.dump(stats, f, indent=2)
    
    def _print_benchmark_summary(self, stats: Dict):
        print("\nBENCHMARK SUMMARY")
        print(f"📊 Evaluated: {stats['total_evaluated']} samples | ✅ Pass Rate: {stats['pass_rate']:.2%} | ❌ Compilation Fail: {stats['compilation_fail_rate']:.2%}")
        print(f"📝 Format: {stats['format']['mean']:.3f} | 🧠 Reasoning: {stats['reasoning']['mean']:.3f} | 💻 Code: {stats['code_quality']['mean']:.3f} | ✓ Correctness: {stats['correctness']['mean']:.3f}")

# ==========================================
# STEP 4: INFERENCE-TIME TRAINING
# ==========================================

class InferenceTimeTraining:
    def __init__(self, model, tokenizer, config: PipelineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def generate_with_self_consistency(self, prompt: str, num_samples: int = 5, temperature: float = 0.8) -> Dict:
        print("\n" + "="*50)
        print("STEP 4: Inference-Time Training (Self-Consistency)")
        print("="*50)
        
        FastLanguageModel.for_inference(self.model)
        
        solutions = []
        for i in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=2048, pad_token_id=self.tokenizer.pad_token_id, temperature=temperature, top_p=0.95, do_sample=True)
            solutions.append("<think>\n" + self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            print(f"Generated solution {i+1}/{num_samples}")
        
        code_solutions = [EnhancedRewardFunctions.extract_code(sol)[1] for sol in solutions if EnhancedRewardFunctions.extract_code(sol)[1]]
        
        if code_solutions:
            from collections import Counter
            best_code, frequency = Counter(code_solutions).most_common(1)[0]
            confidence, best_solution = frequency / len(code_solutions), next((s for s in solutions if EnhancedRewardFunctions.extract_code(s)[1] == best_code), solutions[0])
        else:
            best_solution, confidence = solutions[0], 0.0
            
        print(f"\n✓ Self-consistency confidence: {confidence:.2%}")
        return {'best_solution': best_solution, 'confidence': confidence, 'all_solutions': solutions, 'num_unique_codes': len(set(code_solutions))}
    
    def demonstrate_itt(self, test_problem: Dict):
        result = self.generate_with_self_consistency(f"Think step-by-step inside <think></think> tags, then provide your solution.\n\n{test_problem.get('prompt', test_problem.get('text', ''))}\n\n<think>\n")
        print("\n--- Best Solution (via Self-Consistency) ---")
        print(result['best_solution'][:500] + "...")
        return result

# ==========================================
# MAIN PIPELINE ORCHESTRATOR
# ==========================================

def main():
    print("\n" + "="*70)
    print(" QWEN 3 4B END-TO-END COT TRAINING PIPELINE")
    print("="*70)
    
    config = PipelineConfig()
    
    # ==========================================
    # STEP 1: SFT
    # ==========================================
    sft_pipeline = SFTCoTPipeline(config)
    sft_pipeline.load_model()
    train_dataset, eval_dataset = sft_pipeline.prepare_sft_dataset()
    sft_stats = sft_pipeline.train_sft(train_dataset, eval_dataset)
    
    # 🧹 Critical VRAM Optimization before shifting to GRPO
    print("\n🧹 Cleaning Optimizer States from VRAM...")
    import gc
    sft_pipeline.model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # STEP 2: GRPO
    # ==========================================
    grpo_pipeline = GRPOPipeline(config, sft_pipeline.model, sft_pipeline.tokenizer)
    grpo_dataset = grpo_pipeline.prepare_grpo_dataset()
    grpo_pipeline.train_grpo(grpo_dataset)
    
    # ==========================================
    # STEP 3: BENCHMARKING
    # ==========================================
    benchmark_pipeline = BenchmarkPipeline(sft_pipeline.model, sft_pipeline.tokenizer, config)
    # Pull a localized list from the stream just for the benchmark suite
    test_list = list(islice(grpo_dataset, 50)) 
    benchmark_stats = benchmark_pipeline.run_benchmark(test_list, num_samples=50)
    
    # ==========================================
    # STEP 4: INFERENCE-TIME TRAINING DEMO
    # ==========================================
    itt_pipeline = InferenceTimeTraining(sft_pipeline.model, sft_pipeline.tokenizer, config)
    itt_result = itt_pipeline.demonstrate_itt(test_list[0])
    
    # ==========================================
    # FINAL MODEL SAVE
    # ==========================================
    os.makedirs(config.final_model_dir, exist_ok=True)
    sft_pipeline.model.save_pretrained(config.final_model_dir)
    sft_pipeline.tokenizer.save_pretrained(config.final_model_dir)
    print(f"\n✓ Final model saved to: {config.final_model_dir}")
    print("\n🎉 PIPELINE COMPLETE!")
    
    return {'sft_stats': sft_stats, 'benchmark_stats': benchmark_stats, 'itt_result': itt_result}

if __name__ == "__main__":
    results = main()
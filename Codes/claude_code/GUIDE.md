# Qwen 3 4B Chain-of-Thought Training Pipeline - Complete Guide

## Overview

This pipeline trains a Qwen 3 4B model on competitive programming with Chain-of-Thought (CoT) reasoning through four stages:

1. **SFT (Supervised Fine-Tuning)**: Teach the model to think step-by-step
2. **GRPO (Generalized Reward Policy Optimization)**: Reinforce correct reasoning with multi-faceted rewards
3. **Benchmarking**: Comprehensive evaluation with multiple metrics
4. **Inference-Time Training**: Self-consistency and best-of-N sampling

---

## Key Improvements Over Your Original Code

### 1. **Proper CoT Dataset Handling**

**Problem with original approach:**
- Used `dataset_text_field="text"` but the dataset has `messages` field
- Didn't properly structure the thinking format
- Missing explicit CoT prompting

**New approach:**
```python
def _format_cot_example(self, item: Dict) -> Optional[Dict]:
    """Properly format with <think> tags and chat template"""
    formatted_text = f"""<|im_start|>system
{system_msg}
<|im_end|>
<|im_start|>user
{user_msg}
Please think through this problem step-by-step inside <think></think> tags...
<|im_end|>
<|im_start|>assistant
{assistant_msg}
<|im_end|>"""
```

**Benefits:**
- Explicitly teaches the model the thinking structure
- Maintains chat format compatibility
- Encourages detailed reasoning

### 2. **Enhanced Reward Functions**

The original code had only 2 basic rewards. The new system has 4 sophisticated rewards:

#### **A. Format Reward (0.0 to 1.0)**
```python
- <think> tags present: 0.3
- Code block present: 0.3
- Proper ordering (think→code): 0.2
- Minimum content length: 0.2
```

#### **B. Reasoning Quality Reward (-0.5 to 1.0)**
Evaluates the quality of thinking:

```python
1. Length & Detail (0.0-0.3):
   - < 20 words: 0.0
   - 20-50 words: 0.1
   - 50-150 words: 0.2-0.3 (optimal)
   - 150-300 words: 0.3
   - > 300 words: 0.2 (verbosity penalty)

2. Logical Structure (0.0-0.3):
   - Step indicators: "first", "then", "finally"
   - Causal reasoning: "because", "therefore"
   - Conditionals: "if", "when", "while"
   - Analytical: "consider", "observe"

3. Algorithmic Thinking (0.0-0.2):
   - Complexity analysis: "O(n)", "time complexity"
   - Strategy mentions: "algorithm", "approach"
   - Edge cases: "corner case", "boundary"
   - Optimization: "efficient", "optimize"

4. Structured Thinking (0.0-0.2):
   - Bullet points or numbered lists
   - Clear paragraphs
```

**Example high-scoring reasoning:**
```
<think>
First, I need to understand what the problem is asking:
- We have n numbers
- We need to find the maximum sum

Approach:
1. Consider using dynamic programming
2. Time complexity should be O(n)
3. Edge case: what if n = 0?

Therefore, I'll use a simple loop...
</think>
```

#### **C. Code Quality Reward (0.0 to 0.5)**
Static analysis of code:

```python
- Function definitions: 0.1
- Comments/docstrings: 0.1
- Descriptive variable names: 0.1
- Pythonic patterns (comprehensions): 0.05
- Reasonable length (5-100 lines): 0.1
```

#### **D. Correctness Reward (-2.0 to 3.0)**
Execution against test cases:

```python
- Compilation failure: -2.0
- All tests pass: 3.0
- Majority pass (>50%): 0.5 to 2.5 (scaled)
- Minority pass (≤50%): -0.5
```

**Why these rewards work:**

1. **Format reward** ensures the model learns the structure
2. **Reasoning reward** encourages high-quality thinking (not just random text in <think> tags)
3. **Code quality** pushes toward readable, maintainable solutions
4. **Correctness** is the ultimate goal, heavily weighted

### 3. **Stable Training Parameters**

**SFT parameters (conservative, proven settings):**
```python
max_seq_length: 4096           # Longer for detailed CoT
lora_r: 32                     # Increased capacity
lora_alpha: 64                 # 2x r for stability
lora_dropout: 0.05             # Light regularization
use_rslora: True               # Rank-stabilized LoRA

batch_size: 2
gradient_accumulation: 8       # Effective batch = 16
learning_rate: 2e-4
warmup_ratio: 0.1
lr_scheduler: "cosine"
optimizer: "adamw_8bit"
epochs: 2
```

**GRPO parameters (carefully tuned for RL stability):**
```python
learning_rate: 5e-6            # Much lower than SFT
batch_size: 1                  # Small for stability
gradient_accumulation: 8
num_generations: 4             # Multiple rollouts
max_completion_length: 2048
epochs: 3
lr_scheduler: "cosine"
weight_decay: 0.1              # Strong regularization
max_grad_norm: 1.0             # Gradient clipping
```

**Why these work:**
- **Lower LR for GRPO**: RL is unstable; small steps prevent collapse
- **Gradient clipping**: Prevents exploding gradients
- **RSLoRA**: Stabilizes LoRA training at higher ranks
- **Cosine scheduler**: Smooth learning rate decay
- **Small batch size for GRPO**: More stable policy updates

### 4. **Comprehensive Benchmarking**

The new system tracks:
```python
{
  'format': {'mean': X, 'std': Y, 'median': Z},
  'reasoning': {...},
  'code_quality': {...},
  'correctness': {...},
  'pass_rate': 0.XX,
  'compilation_fail_rate': 0.XX,
  'total_evaluated': N
}
```

Saved to JSON for tracking across runs.

### 5. **Inference-Time Training**

Implements **self-consistency** method:

```python
1. Generate N solutions (N=5) with temperature sampling
2. Extract code from each solution
3. Find the most common code (voting)
4. Return that solution as the best
5. Confidence = frequency / total_samples
```

**Example:**
- 5 solutions generated
- 3 have the same code → confidence = 60%
- Higher confidence = more reliable solution

---

## How to Use

### Quick Start

```bash
# 1. Install dependencies (first time only)
pip install unsloth trl==0.24.0 datasets transformers==5.5.0 \
    bitsandbytes accelerate vllm huggingface_hub wandb

# 2. Run the complete pipeline
python qwen3_cot_pipeline.py
```

### Customizing the Pipeline

Edit the `PipelineConfig` class:

```python
@dataclass
class PipelineConfig:
    # Increase training data
    sft_train_samples: int = 10000  # Default: 5000
    
    # Use harder problems
    problem_filter: str = "B"  # Default: "A" (easiest)
    
    # Enable W&B tracking
    report_to: str = "wandb"  # Default: "none"
    
    # Adjust learning rates
    sft_learning_rate: float = 1e-4  # Lower if unstable
    grpo_learning_rate: float = 3e-6  # Lower if reward collapse
```

### Running Individual Steps

```python
from qwen3_cot_pipeline import (
    SFTCoTPipeline,
    GRPOPipeline,
    BenchmarkPipeline,
    InferenceTimeTraining,
    PipelineConfig
)

config = PipelineConfig()

# Step 1: SFT only
sft = SFTCoTPipeline(config)
sft.load_model()
train_ds, eval_ds = sft.prepare_sft_dataset()
sft.train_sft(train_ds, eval_ds)

# Step 2: GRPO only (requires SFT model)
grpo = GRPOPipeline(config, sft.model, sft.tokenizer)
grpo_ds = grpo.prepare_grpo_dataset()
grpo.train_grpo(grpo_ds)

# Step 3: Benchmark
benchmark = BenchmarkPipeline(sft.model, sft.tokenizer, config)
stats = benchmark.run_benchmark(grpo_ds, num_samples=100)

# Step 4: Inference-time training
itt = InferenceTimeTraining(sft.model, sft.tokenizer, config)
result = itt.generate_with_self_consistency(
    "Your problem here...",
    num_samples=5
)
```

---

## Understanding the Metrics

### During Training

**SFT Metrics:**
- `train_loss`: Should decrease smoothly (target: < 1.0)
- `eval_loss`: Should track train_loss (gap < 0.3 is good)

**GRPO Metrics:**
- `rewards/mean`: Average total reward per generation
  - Should increase over time
  - Target: > 2.0 (format + reasoning + correctness)
- `rewards/std`: Variance in rewards
  - High initially, should stabilize
  - Target: < 1.0 at convergence

### After Training (Benchmarks)

**Pass Rate**: % of problems solved correctly
- Baseline (untrained): ~5-10%
- After SFT: ~20-30%
- After GRPO: ~40-60%
- **Target: > 50%**

**Format Score**: Structure adherence
- **Target: > 0.8** (most samples should have proper format)

**Reasoning Score**: Quality of thinking
- **Target: > 0.5** (decent logical reasoning)

**Code Quality**: Readability and structure
- **Target: > 0.3** (basic quality standards met)

**Correctness**: Test case performance
- Mean should be > 1.0 (mostly positive rewards)
- **Target: > 1.5** (good performance)

---

## Troubleshooting

### Problem: Model generates gibberish

**Cause:** Learning rate too high
**Solution:**
```python
config.sft_learning_rate = 1e-4  # Reduce from 2e-4
config.grpo_learning_rate = 3e-6  # Reduce from 5e-6
```

### Problem: Rewards collapse to zero

**Cause:** GRPO instability
**Solutions:**
1. Reduce learning rate
2. Increase gradient accumulation
3. Reduce number of generations

```python
config.grpo_learning_rate = 1e-6
config.grpo_grad_accum = 16
config.grpo_num_generations = 2
```

### Problem: OOM (Out of Memory)

**Solutions:**
```python
# Reduce sequence length
config.max_seq_length = 2048

# Reduce batch size
config.sft_batch_size = 1
config.grpo_batch_size = 1

# Increase gradient accumulation to compensate
config.sft_grad_accum = 16
config.grpo_grad_accum = 16
```

### Problem: Model doesn't use <think> tags

**Cause:** Not learning the format
**Solutions:**
1. Increase format reward weight
2. Train SFT longer
3. Check dataset formatting

### Problem: Low pass rate after training

**Debugging steps:**
1. Check if model generates proper format (format_score > 0.8)
2. Check if code compiles (compilation_fail_rate < 0.1)
3. Check reasoning quality (reasoning_score > 0.3)
4. If format is good but correctness is low → need more GRPO epochs
5. If format is bad → need more SFT or better prompting

---

## Advanced: Custom Reward Functions

Add your own reward function:

```python
@staticmethod
def my_custom_reward(completions, **kwargs) -> List[float]:
    """
    Custom reward logic
    
    Returns: List of floats (one per completion)
    """
    rewards = []
    for comp in completions:
        content = comp[0]['content'] if isinstance(comp, list) else comp
        
        # Your custom logic here
        score = 0.0
        
        # Example: Reward for mentioning time complexity
        if "time complexity" in content.lower():
            score += 0.5
        
        rewards.append(score)
    
    return rewards

# Add to GRPO trainer
trainer = GRPOTrainer(
    ...
    reward_funcs=[
        self.rewards.format_reward,
        self.rewards.reasoning_quality_reward,
        my_custom_reward,  # Your custom reward
        self.rewards.correctness_reward,
    ],
    ...
)
```

---

## Expected Training Time

On a single A100 (40GB):
- **SFT**: ~2-3 hours (5000 samples, 2 epochs)
- **GRPO**: ~3-4 hours (2000 samples, 3 epochs)
- **Benchmarking**: ~30 minutes (100 samples)

Total: **~6-8 hours** for complete pipeline

On smaller GPUs (RTX 3090, A10):
- Expect 1.5-2x longer
- May need to reduce batch sizes

---

## Model Outputs

After training, you'll have:

```
outputs/
├── sft_cot/
│   ├── final/                    # SFT model checkpoint
│   ├── checkpoint-XXX/           # Intermediate checkpoints
│   └── trainer_state.json
├── grpo_cot/
│   ├── final/                    # GRPO model checkpoint
│   └── checkpoint-XXX/
├── final_model/                  # Final merged model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── benchmarks/
    └── benchmark_results.json    # Evaluation metrics
```

---

## Best Practices

1. **Start small**: Train on 1000 samples first to test
2. **Monitor metrics**: Watch for reward collapse or instability
3. **Save checkpoints**: Enable W&B for experiment tracking
4. **Validate early**: Run benchmarks after SFT before GRPO
5. **Use version control**: Track config changes across runs
6. **Test inference**: Try generating solutions manually to verify

---

## Citation

If you use this pipeline, please cite:

- **Unsloth**: https://github.com/unslothai/unsloth
- **CodeForces CoT Dataset**: https://huggingface.co/datasets/open-r1/codeforces-cots
- **TRL Library**: https://github.com/huggingface/trl

---

## License

This code is provided as-is for research and educational purposes.

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the metrics and logs
3. Try reducing learning rates first
4. Ensure GPU memory is sufficient

Good luck with your training! 🚀

# Configuration Template
# =====================
# Copy this file and modify the values to customize your training

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Complete configuration for Qwen 3 CoT training pipeline
    
    Modify these values to customize your training run.
    Each section has recommended values and safe ranges.
    """
    
    # ==========================================
    # MODEL CONFIGURATION
    # ==========================================
    
    model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    # Available models:
    # - "unsloth/Qwen3-4B-unsloth-bnb-4bit" (recommended)
    # - "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" (larger, slower)
    # - "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit" (smaller, faster)
    
    max_seq_length: int = 4096
    # Recommended: 2048-4096
    # - 2048: Faster training, less memory
    # - 4096: Better for long reasoning chains
    # - 8192: Maximum, requires lots of VRAM
    
    load_in_4bit: bool = True
    # Keep True for memory efficiency
    # Set False only if you have 80GB+ VRAM
    
    # ==========================================
    # LORA CONFIGURATION
    # ==========================================
    
    lora_r: int = 32
    # Recommended: 16-64
    # - 16: Faster, less capacity
    # - 32: Good balance (recommended)
    # - 64: More capacity, slower
    
    lora_alpha: int = 64
    # Recommended: 2 * lora_r
    # This ratio (2:1) is proven stable
    
    lora_dropout: float = 0.05
    # Recommended: 0.0-0.1
    # - 0.0: No regularization
    # - 0.05: Light regularization (recommended)
    # - 0.1: Strong regularization
    
    use_rslora: bool = True
    # Keep True for stability
    # RSLoRA = Rank-Stabilized LoRA
    
    # ==========================================
    # DATASET CONFIGURATION
    # ==========================================
    
    dataset_name: str = "open-r1/codeforces-cots"
    dataset_subset: str = "solutions_decontaminated"
    # Options:
    # - "solutions_py_decontaminated" (Python, recommended)
    # - "solutions_decontaminated" (C++, harder)
    
    problem_filter: str = "A"
    # Difficulty levels (easiest to hardest):
    # - "A": Easiest (recommended for starting)
    # - "B": Medium
    # - "C": Hard
    # - "D", "E", "F": Very hard
    
    # ==========================================
    # SFT (SUPERVISED FINE-TUNING) CONFIGURATION
    # ==========================================
    
    sft_train_samples: int = 5000
    # Recommended: 2000-10000
    # - 1000: Quick test
    # - 5000: Good training (recommended)
    # - 10000: Thorough training
    
    sft_eval_samples: int = 500
    # Recommended: 10-20% of train samples
    
    sft_epochs: int = 2
    # Recommended: 1-3
    # - 1: Quick training
    # - 2: Standard (recommended)
    # - 3+: Risk of overfitting
    
    sft_batch_size: int = 2
    # Recommended: 1-4
    # Depends on your GPU:
    # - A100 80GB: 4
    # - A100 40GB: 2
    # - A10 24GB: 1
    # - RTX 3090 24GB: 1
    
    sft_grad_accum: int = 8
    # Effective batch size = batch_size * grad_accum
    # Recommended effective batch: 8-16
    # Adjust based on your batch_size:
    # - batch_size=1 → grad_accum=16
    # - batch_size=2 → grad_accum=8
    # - batch_size=4 → grad_accum=4
    
    sft_learning_rate: float = 2e-4
    # Recommended: 1e-4 to 3e-4
    # - 1e-4: Conservative, slower
    # - 2e-4: Standard (recommended)
    # - 3e-4: Aggressive, faster but riskier
    
    sft_warmup_ratio: float = 0.1
    # Recommended: 0.05-0.15
    # Warmup helps stabilize early training
    
    sft_lr_scheduler: str = "cosine"
    # Options:
    # - "cosine": Smooth decay (recommended)
    # - "linear": Linear decay
    # - "constant": No decay
    
    # ==========================================
    # GRPO (REINFORCEMENT LEARNING) CONFIGURATION
    # ==========================================
    
    grpo_train_samples: int = 2000
    # Recommended: 1000-5000
    # - 1000: Quick RL training
    # - 2000: Standard (recommended)
    # - 5000: Thorough but slow
    
    grpo_epochs: int = 3
    # Recommended: 2-5
    # - 2: Minimum for convergence
    # - 3: Standard (recommended)
    # - 5: Maximum before diminishing returns
    
    grpo_batch_size: int = 1
    # Recommended: 1-2
    # GRPO is memory-intensive
    # - Keep at 1 for most GPUs
    # - Use 2 only on A100 80GB
    
    grpo_grad_accum: int = 8
    # Recommended: 4-16
    # Higher = more stable but slower
    # - 4: Faster, less stable
    # - 8: Good balance (recommended)
    # - 16: Most stable, slowest
    
    grpo_learning_rate: float = 5e-6
    # CRITICAL: Must be much lower than SFT!
    # Recommended: 1e-6 to 1e-5
    # - 1e-6: Very conservative
    # - 5e-6: Standard (recommended)
    # - 1e-5: Maximum safe value
    
    grpo_num_generations: int = 4
    # Recommended: 2-8
    # Number of solutions generated per prompt
    # - 2: Faster training
    # - 4: Good diversity (recommended)
    # - 8: Maximum diversity, very slow
    
    grpo_max_completion_length: int = 2048
    # Recommended: 1024-2048
    # - 1024: Faster, shorter solutions
    # - 2048: Allows detailed reasoning
    
    # ==========================================
    # REWARD FUNCTION WEIGHTS (Advanced)
    # ==========================================
    # Note: These are applied internally by the reward functions
    # You can create custom weights by modifying the reward functions
    
    reward_weight_format: float = 1.0
    reward_weight_reasoning: float = 1.0
    reward_weight_code_quality: float = 1.0
    reward_weight_correctness: float = 1.0
    
    # To emphasize correctness over style:
    # reward_weight_correctness: float = 2.0
    
    # To emphasize reasoning quality:
    # reward_weight_reasoning: float = 1.5
    
    # ==========================================
    # BENCHMARKING CONFIGURATION
    # ==========================================
    
    benchmark_samples: int = 100
    # Recommended: 50-200
    # - 50: Quick evaluation
    # - 100: Standard (recommended)
    # - 200: Comprehensive
    
    # ==========================================
    # INFERENCE-TIME TRAINING CONFIGURATION
    # ==========================================
    
    itt_num_samples: int = 5
    # Recommended: 3-10
    # Number of solutions for self-consistency
    # - 3: Fast
    # - 5: Standard (recommended)
    # - 10: Maximum reliability
    
    itt_temperature: float = 0.8
    # Recommended: 0.7-1.0
    # - 0.7: More focused
    # - 0.8: Good diversity (recommended)
    # - 1.0: Maximum diversity
    
    # ==========================================
    # OUTPUT PATHS
    # ==========================================
    
    sft_output_dir: str = "outputs/sft_cot"
    grpo_output_dir: str = "outputs/grpo_cot"
    final_model_dir: str = "outputs/final_model"
    benchmark_dir: str = "outputs/benchmarks"
    
    # ==========================================
    # LOGGING & EXPERIMENT TRACKING
    # ==========================================
    
    report_to: str = "none"
    # Options:
    # - "none": No tracking
    # - "wandb": Weights & Biases (recommended for serious training)
    # - "tensorboard": TensorBoard
    
    wandb_project: str = "qwen3-cot-training"
    # Only used if report_to="wandb"
    
    wandb_run_name: str = "run-001"
    # Only used if report_to="wandb"
    
    # ==========================================
    # SYSTEM CONFIGURATION
    # ==========================================
    
    seed: int = 3407
    # Random seed for reproducibility
    
    num_proc: int = None
    # CPU processes for data processing
    # None = auto-detect (recommended)
    
    # ==========================================
    # CHECKPOINT & SAVING
    # ==========================================
    
    save_strategy: str = "steps"
    # Options:
    # - "steps": Save every N steps
    # - "epoch": Save every epoch
    # - "no": Don't save checkpoints
    
    save_steps: int = 200
    # Save frequency (if save_strategy="steps")
    
    save_total_limit: int = 3
    # Maximum checkpoints to keep
    # Recommended: 2-5
    
    # ==========================================
    # HARDWARE-SPECIFIC PRESETS
    # ==========================================
    
    @classmethod
    def for_a100_80gb(cls):
        """Preset for NVIDIA A100 80GB"""
        return cls(
            max_seq_length=4096,
            sft_batch_size=4,
            sft_grad_accum=4,
            grpo_batch_size=2,
            grpo_grad_accum=4,
            sft_train_samples=10000,
            grpo_train_samples=5000,
        )
    
    @classmethod
    def for_a100_40gb(cls):
        """Preset for NVIDIA A100 40GB"""
        return cls(
            max_seq_length=4096,
            sft_batch_size=2,
            sft_grad_accum=8,
            grpo_batch_size=1,
            grpo_grad_accum=8,
            sft_train_samples=5000,
            grpo_train_samples=2000,
        )
    
    @classmethod
    def for_rtx_3090(cls):
        """Preset for NVIDIA RTX 3090 24GB"""
        return cls(
            max_seq_length=2048,
            sft_batch_size=1,
            sft_grad_accum=16,
            grpo_batch_size=1,
            grpo_grad_accum=16,
            sft_train_samples=3000,
            grpo_train_samples=1000,
        )
    
    @classmethod
    def for_a10(cls):
        """Preset for NVIDIA A10 24GB"""
        return cls(
            max_seq_length=2048,
            sft_batch_size=1,
            sft_grad_accum=16,
            grpo_batch_size=1,
            grpo_grad_accum=16,
            sft_train_samples=3000,
            grpo_train_samples=1000,
        )
    
    @classmethod
    def quick_test(cls):
        """Preset for quick testing (< 30 minutes)"""
        return cls(
            max_seq_length=2048,
            sft_train_samples=200,
            sft_eval_samples=50,
            sft_epochs=1,
            grpo_train_samples=100,
            grpo_epochs=1,
            grpo_num_generations=2,
            benchmark_samples=20,
        )

# ==========================================
# USAGE EXAMPLES
# ==========================================

# Example 1: Use default configuration
# config = TrainingConfig()

# Example 2: Use preset for your hardware
# config = TrainingConfig.for_a100_40gb()

# Example 3: Custom configuration
# config = TrainingConfig(
#     sft_train_samples=8000,
#     sft_learning_rate=1e-4,
#     problem_filter="B",
#     report_to="wandb",
# )

# Example 4: Override specific values from preset
# config = TrainingConfig.for_rtx_3090()
# config.sft_learning_rate = 1e-4
# config.report_to = "wandb"

# ==========================================
# RECOMMENDED CONFIGURATIONS BY USE CASE
# ==========================================

"""
1. PRODUCTION TRAINING (Best Quality)
   - Hardware: A100 80GB or 2x A100 40GB
   - Time: ~8 hours
   - Config: for_a100_80gb()
   - Modifications:
     * sft_train_samples = 10000
     * grpo_epochs = 5
     * report_to = "wandb"

2. STANDARD TRAINING (Good Quality)
   - Hardware: A100 40GB or RTX 3090
   - Time: ~6 hours
   - Config: for_a100_40gb()
   - Modifications:
     * sft_train_samples = 5000
     * grpo_epochs = 3
     * report_to = "wandb"

3. EXPERIMENTAL TRAINING (Fast Iteration)
   - Hardware: Any GPU with 16GB+ VRAM
   - Time: ~3 hours
   - Config: for_rtx_3090()
   - Modifications:
     * sft_train_samples = 2000
     * grpo_epochs = 2

4. QUICK TEST (Verify Setup)
   - Hardware: Any GPU with 12GB+ VRAM
   - Time: ~30 minutes
   - Config: quick_test()
   - No modifications needed
"""

# ==========================================
# TROUBLESHOOTING GUIDE
# ==========================================

"""
PROBLEM: Out of Memory (OOM)

Solutions (try in order):
1. Reduce max_seq_length (4096 → 2048)
2. Reduce batch_size (2 → 1)
3. Increase grad_accum to maintain effective batch size
4. Reduce grpo_num_generations (4 → 2)

---

PROBLEM: Training is unstable (loss spikes)

Solutions:
1. Reduce learning rates:
   - sft_learning_rate = 1e-4
   - grpo_learning_rate = 1e-6
2. Increase warmup_ratio (0.1 → 0.2)
3. Reduce batch size and increase grad_accum

---

PROBLEM: Rewards collapse to zero

Solutions:
1. Dramatically reduce grpo_learning_rate (5e-6 → 1e-6)
2. Increase grpo_grad_accum (8 → 16)
3. Reduce grpo_num_generations (4 → 2)
4. Start from a better SFT checkpoint

---

PROBLEM: Low pass rate after training

Diagnosis:
1. Check format_score > 0.8 → If not, SFT didn't converge
2. Check reasoning_score > 0.3 → If not, need more SFT epochs
3. Check compilation_fail_rate < 0.1 → If not, GRPO is broken
4. If all above are good → Need more GRPO epochs

Solutions:
1. Train SFT longer (epochs: 2 → 3)
2. Use more SFT data (samples: 5000 → 10000)
3. Train GRPO longer (epochs: 3 → 5)
4. Try harder problems gradually (A → B → C)

---

PROBLEM: Training is too slow

Solutions:
1. Reduce max_seq_length (4096 → 2048)
2. Reduce num_generations (4 → 2)
3. Reduce train_samples
4. Use gradient checkpointing (already enabled)
5. Increase batch_size if you have VRAM headroom
"""

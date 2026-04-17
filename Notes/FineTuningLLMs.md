# Fine Tuning LLMs guide
Guide URL : https://unsloth.ai/docs/get-started/fine-tuning-llms-guide


## What is fine tuning 

Standard method is sft called supervised fine tuning. Other methods are grpo gspo where an agent learns to make decisions by interacting with an environment and receiving feedback int the form of rewards or penalties


Unsloth notebooks for fine tuning the models on 3 gb vram
https://unsloth.ai/docs/get-started/unsloth-notebooks


- Update + Learn New Knowledge: Inject and learn new domain-specific information.

- Customize Behavior: Adjust the model’s tone, personality, or response style.

- Optimize for Tasks: Improve accuracy and relevance for specific use cases.

## What is LoRA and QLoRA ?? 
LoRA = low rank adaptation
Q LoRA = Quantized Low Rank adaptation

In LLMs, we have model weights. Llama 70B has 70 billion numbers. Instead of changing all 70B numbers, we instead add thin matrices A and B to each weight, and optimize those. This means we only optimize 1% of weights. LoRA is when the original model is 16-bit unquantized while QLoRA quantizes to 4-bit to save 75% memory.

## Choose the right model and the perfect method


## Instruct vs normal model what is the difference between them 


What model should i use for fine tuning ??
- choose a model that align with your usecase
- Assess your storage compute capacity and dataset
- select a model and parameters
- choose between base and isntruct model 


### Instruct models 
Instruct models are pre-trained with built-in instructions, making them ready to use without any fine-tuning. These models, including GGUFs and others commonly available, are optimized for direct usage and respond effectively to prompts right out of the box. Instruct models work with conversational chat templates like ChatML or ShareGPT.

### Base Models 
Base models, on the other hand, are the original pre-trained versions without instruction fine-tuning. These are specifically designed for customization through fine-tuning, allowing you to adapt them to your unique needs. Base models are compatible with instruction-style templates like Alpaca or Vicuna, but they generally do not support conversational chat templates out of the box.


### Should i chose instruct or base ??
The decision often depends on the quantity, quality, and type of your data:

- 1,000+ Rows of Data: If you have a large dataset with over 1,000 rows, it's generally best to fine-tune the base model.

- 300–1,000 Rows of High-Quality Data: With a medium-sized, high-quality dataset, fine-tuning the base or instruct model are both viable options.

- Less than 300 Rows: For smaller datasets, the instruct model is typically the better choice. Fine-tuning the instruct model enables it to align with specific needs while preserving its built-in instructional capabilities. This ensures it can follow general instructions without additional input unless you intend to significantly alter its functionality.

- For information how how big your dataset should be, see here


## Unsloth dynamic 4 bit quantization
See docs to actually see the available options here ..
Dynamically selects what params to quantize and what not to ..




# LoRA fine-tuning hyperparams guide 

LoRA hyperparameters are tunable settings that govern how Low-Rank Adaptation fine-tunes LLMs. With many choices (e.g., learning rate and epochs) and countless combinations, picking the right values is key to accuracy, stability, quality, and fewer hallucinations. Done well, LoRA can match full fine-tuning performance while using 4× less VRAM.



Read this : https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide

Target Modules

- Specify which parts of the model you want to apply LoRA adapters to — either the attention, the MLP, or both.
- Attention: q_proj, k_proj, v_proj, o_proj
- MLP: gate_proj, up_proj, down_proj
- Recommended to target all major linear layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.


# Reinforcement learning guide 

https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide 




# Learning about GRPO and ho wto implement it for our qwen 3 billion param model



We may need to mix the datasets like deepseek did 
Can add tools like let the model executes its own code and check the test cases if any of them fails it redoes it .. may be help ful



https://www.youtube.com/watch?v=xT4jxQUl0X8




## Reinforce

Understanding the REINFORCE algorithm is the perfect stepping stone to Proximal Policy Optimization (PPO). 

If you've spent time with algorithms like Q-Learning or DQN, you are used to **Value-Based** reinforcement learning. In those, the neural network learns the *value* of being in a state and taking an action, and the policy is just "pick the action with the highest Q-value."

REINFORCE flips this completely. It is a **Policy-Based** method (specifically, a Policy Gradient method). Instead of calculating values, the neural network directly outputs the probability of taking each action. 

Here is a breakdown of how REINFORCE works, the math behind it, and why it eventually necessitates the upgrades found in PPO.

### 1. The Objective Function
In policy gradient methods, our neural network has parameters $\theta$ that define our policy, written as $\pi_\theta(a|s)$. This represents the probability of taking action $a$ in state $s$.

Our goal is to find the parameters $\theta$ that maximize the expected return (total accumulated reward) of a trajectory $\tau$. Let's call this objective function $J(\theta)$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

To maximize this, we need to perform gradient ascent. We need to find $\nabla_\theta J(\theta)$ and update our network weights in that direction.

### 2. The Policy Gradient Theorem
Taking the gradient of an expectation that depends on the very parameters we are tweaking is mathematically tricky. Skipping the heavy calculus derivation, the **Policy Gradient Theorem** elegantly simplifies this to:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) G_t \right]$$

Let's break down this formula intuitively, as it is the absolute heart of REINFORCE:
* $\pi_\theta(a_t | s_t)$: The probability your network assigned to the action it actually took.
* $\nabla_\theta \log \pi_\theta(a_t | s_t)$: The direction to tweak the network weights to make that specific action *more likely* in the future.
* $G_t$: The actual Return (cumulative reward) received from time step $t$ onwards.

**The core intuition:** If an action resulted in a high positive return ($G_t > 0$), the gradient pushes the network to increase the probability of that action. If the action resulted in a negative return ($G_t < 0$), it pushes the network to decrease that probability. The $\log$ term ensures we don't disproportionately update actions that already have a high probability.

### 3. The REINFORCE Algorithm (Monte Carlo Policy Gradient)
REINFORCE is simply applying this math using full trajectories (Monte Carlo sampling). 

1.  **Initialize** the policy network with random weights $\theta$.
2.  **Collect a Trajectory:** Use the current policy $\pi_\theta$ to play an entire episode from start to finish, recording states, actions, and rewards: $s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T$.
3.  **Calculate Returns:** For every time step $t$ in that episode, calculate the total discounted return from that point forward: $G_t = \sum_{k=t+1}^T \gamma^{k-t-1} r_k$
4.  **Compute the Gradient and Update:** Loop through each time step $t$ in the trajectory and update the network parameters:
    $$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t | s_t) G_t$$
    *(where $\alpha$ is the learning rate)*
5.  **Repeat** steps 2-4 until the policy converges.

Here is an interactive visualization of how a single gradient update shifts the probabilities of a policy based on the return $G_t$.

```json?chameleon
{"component":"LlmGeneratedComponent","props":{"height":"700px","prompt":"Objective: Create an interactive Policy Gradient (REINFORCE) visualizer for a continuous 1D action space.\nData State: initialValues = { mean: 0, std: 1, sampledAction: 1.5, G: 5, lr: 0.1 }.\nStrategy: Standard Layout.\nInputs: Provide a slider for 'Return (G)' ranging from -20 to +20. Provide a slider for 'Learning Rate' ranging from 0.01 to 0.5. Provide a button to 'Sample New Action'.\nBehavior: Display a D3.js line chart showing a Gaussian distribution curve representing the initial policy pi_theta (mean=0, std=1) with a dashed line. Plot the 'sampledAction' as a distinct point on the x-axis. Calculate the updated mean based on the REINFORCE update rule: new_mean = old_mean + lr * G * (sampledAction - old_mean) / (std^2). Dynamically draw the updated policy distribution as a solid line curve. Display the mathematical update rule dynamically as values change. When positive returns are applied, visually demonstrate the curve shifting towards the sampled action; for negative returns, demonstrate it shifting away. Include an axis and labels.","id":"im_cd458acaaa9550f5"}}
```

### 4. Why REINFORCE needs an upgrade (The path to PPO)
If you code REINFORCE, you will quickly notice it is highly unstable. This is due to **High Variance**. 

Because $G_t$ is calculated from a single randomly sampled trajectory, playing the exact same game twice might yield wildly different $G_t$ values due to chance. One lucky critical hit might make the algorithm think a terrible action was actually genius. Furthermore, updating the policy purely based on these wild swings can destroy a good policy in a single training step.

This is exactly what PPO solves. To fix the variance of $G_t$, we introduce an Actor-Critic architecture. To fix the destructive policy updates, PPO "clips" the update so the network can't change its probabilities too drastically in one step.

To continue bridging the gap to PPO, would you like to explore how adding a "Value Baseline" to this math stabilizes the variance, or would you prefer to jump straight into how PPO restricts the size of the policy updates?









It is incredibly satisfying to build up to PPO this way. Since you are already familiar with the instability of REINFORCE, understanding the specific mitigations added step-by-step makes PPO’s seemingly weird math completely logical. 

We are going to solve two major problems with REINFORCE: **High Variance** (which makes learning chaotic) and **Destructive Updates** (which can ruin a good policy in a single training step).

Here is the step-by-step evolution from REINFORCE to Proximal Policy Optimization (PPO).

### Step 1: Solving Variance with a "Critic" (Enter A2C)

**The Problem:** In REINFORCE, we multiply our log probability by the absolute return, $G_t$. If you are playing a game where every action gives a reward between +1000 and +1010, REINFORCE will try to increase the probability of *every* action taken, even the one that resulted in the worst possible score of +1000. It lacks a frame of reference.

**The Mitigation:** We introduce a baseline. Instead of asking "How much total reward did we get?", we ask, **"How much *better* was this action than what we expected?"**

To do this, we add a second neural network called the **Critic** (or Value Network), $V_\phi(s)$. Its only job is to look at a state and predict the expected total return from that point on.

Now, instead of using $G_t$, we calculate the **Advantage**, $A_t$:
$$A_t = G_t - V_\phi(s_t)$$

* If $A_t > 0$: The action was better than expected. Increase its probability.
* If $A_t < 0$: The action was worse than expected. Decrease its probability.

**The New Objective:** This gives us the Advantage Actor-Critic (A2C) update rule:
$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) A_t \right]$$

### Step 2: Solving Destructive Updates (The Trust Region)

**The Problem:** Even with the Advantage function stabilizing things, neural networks are sensitive. If we get a really high Advantage ($A_t$), the gradient step might be massive. In standard Deep Learning, a bad step just means your loss temporarily goes up. In Reinforcement Learning, if your policy takes a bad step and falls off a "cliff," it starts generating garbage data. Because it learns from its own data, it might never recover.

**The Mitigation:** We need to limit how much the policy can change in a single update. To do this, we stop looking at log probabilities and start looking at the **Probability Ratio** between the *new* policy and the *old* policy:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

* If $r_t > 1$: The action is more likely now than it was before the update.
* If $r_t < 1$: The action is less likely.

Algorithms like TRPO (Trust Region Policy Optimization) use this ratio and apply complex math (KL Divergence) to physically constrain the network so $r_t$ stays very close to $1.0$. However, TRPO is computationally expensive and difficult to implement.

### Step 3: The PPO Simplification (The Clip)

**The Problem:** TRPO's math is too slow. We want the safety of limiting the policy update without the computational nightmare of calculating second-order derivatives.

**The Mitigation (The Birth of PPO):** PPO's genius is that it achieves the same goal as TRPO but does it with a simple mathematical "clip". We introduce a hyperparameter $\epsilon$ (epsilon), typically set to $0.2$. We define a "trust region" between $1 - \epsilon$ (0.8) and $1 + \epsilon$ (1.2).

If the ratio $r_t$ tries to move outside this region, PPO mathematically flattens the objective function so the gradient becomes zero, and the network stops updating for that specific action.

Here is the famous **PPO Clipped Surrogate Objective**:
$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

Let's visualize exactly how this clipping behaves based on whether the action was good or bad. 

```json?chameleon
{"component":"LlmGeneratedComponent","props":{"height":"600px","prompt":"Objective: Visualize the PPO Clipped Surrogate Objective function to demonstrate how policy updates are restricted.\nData State: initialValues = { advantage: 1.0, epsilon: 0.2 }.\nStrategy: Standard Layout.\nInputs: Provide a slider for 'Advantage (A)' ranging from -3.0 to 3.0. Provide a slider for 'Epsilon' ranging from 0.05 to 0.50.\nBehavior: Display a D3.js line chart where the X-axis is the Probability Ratio (r_t) from 0.0 to 2.0, and the Y-axis is the Objective Value (L). Plot two lines: a dashed line representing the unclipped objective (y = r_t * Advantage) and a thick solid line representing the clipped objective (y = min(r_t * Advantage, clip(r_t, 1-epsilon, 1+epsilon) * Advantage)). Highlight the 'Trust Region' visually on the x-axis between (1-epsilon) and (1+epsilon). Dynamically update the graph and the shape of the clipped line as the Advantage and Epsilon sliders change. Observe how a negative advantage completely flips the shape of the clipping constraint compared to a positive advantage. Add clear axis labels and a dynamic text readout explaining the current state (e.g., 'Advantage is positive: The policy is rewarded for increasing the action probability, but the reward is capped at r_t = 1+epsilon').","id":"im_970b14c26c2d3e2d"}}
```

### Summary of the PPO Logic:

1.  **If Advantage is Positive ($A > 0$):** The action was good. We want $r_t$ to increase. However, if $r_t$ goes above $1 + \epsilon$ (e.g., 1.2), the clip engages. PPO says, *"Okay, you made the action 20% more likely. That's enough for one update. Stop."*
2.  **If Advantage is Negative ($A < 0$):** The action was bad. We want $r_t$ to decrease. If $r_t$ drops below $1 - \epsilon$ (e.g., 0.8), the clip engages. PPO says, *"You made the action 20% less likely. Don't push it completely to zero just yet, let's wait for the next batch of data."*

By taking the minimum of the clipped and unclipped versions, PPO creates a pessimistic lower bound. It ensures the algorithm never gets overly optimistic about a massive, potentially destructive policy update.







You have officially arrived at the cutting edge. Transitioning from PPO to **GRPO (Group Relative Policy Optimization)** is exactly what top AI labs—most notably DeepSeek, who used it to train DeepSeek-R1—are doing right now.

To understand why GRPO exists, we first need to look at the massive, hidden cost of PPO when applied to Large Language Models (LLMs).

### The Problem with PPO: The "Critic Tax"
In the last step, we fixed REINFORCE’s variance by introducing a **Critic** (Value Network) to calculate the Advantage ($A_t = G_t - V_\phi(s_t)$). 

When you are playing an Atari game, a Critic network is tiny and cheap. But when you are training a massive LLM (like a 7B or 70B parameter model), the Critic network is usually *another* LLM of the exact same size. 

To train an LLM with PPO, your GPU memory has to hold:
1.  The **Actor Model** (The policy you are training)
2.  The **Critic Model** (The value network predicting the baseline)
3.  The **Reference Model** (To keep the model from drifting too far)
4.  The **Reward Model** (To score the outputs)

This requires an obscene amount of VRAM. The Critic model alone effectively doubles your memory requirements and compute costs, just to calculate a baseline. 

**GRPO asks a simple question:** *Can we get the stability of an Advantage function without having to pay for a massive Critic network?*

Here is the step-by-step transition to GRPO.

### Step 1: The "Group" (Ditching the Critic)
Instead of asking a Critic network, "What score do you *expect* this state to get?", GRPO takes a purely empirical approach. 

For a single state (or a single prompt, like *"Write a Python script for..."*), GRPO asks the Actor model to generate a **Group** of $G$ different outputs (e.g., $G = 4$ or $G = 8$ different generated scripts). 

### Step 2: The "Relative" Advantage
Next, we take those $G$ outputs and score them using our Reward Model (or a rule-based verifier, like a compiler checking if the code runs). Let's say we get $G$ raw rewards: $\{r_1, r_2, ..., r_G\}$.

Now, we calculate the Advantage purely relative to the group itself using standard Z-score normalization. 

First, calculate the mean ($\mu$) and standard deviation ($\sigma$) of the group's rewards. Then, the Advantage $A_i$ for each specific output $i$ is:

$$A_i = \frac{r_i - \mu}{\sigma}$$

* If an output scored **better than the group average**, it gets a positive Advantage ($A_i > 0$).
* If an output scored **worse than the group average**, it gets a negative Advantage ($A_i < 0$).

By comparing the outputs against each other, **we completely eliminate the need for a Critic network.** The group *is* the baseline.

Let's visualize exactly how raw rewards translate into Relative Advantages within a group.

```json?chameleon
{"component":"LlmGeneratedComponent","props":{"height":"650px","prompt":"Objective: Visualize how GRPO calculates Group Relative Advantage without a Critic network.\nData State: initialValues = { r1: 5, r2: 7, r3: 4, r4: 9, r5: 2 }.\nStrategy: Standard Layout.\nInputs: Provide 5 sliders for 'Output 1 Reward' through 'Output 5 Reward', ranging from 0 to 10.\nBehavior: Display two side-by-side or stacked D3.js bar charts. The first chart shows the 'Raw Rewards' for the 5 outputs as standard bars. Overlay a dynamic, horizontal dashed line across this chart representing the calculated Mean of the 5 rewards. The second chart displays the calculated 'Relative Advantage (A_i)' for each output using the formula: A_i = (r_i - Mean) / StdDev. Color the bars in the second chart green if positive, and red if negative. Below the charts, dynamically display the mathematical text showing the current Mean, Standard Deviation, and the explicit calculation for Output 1's Advantage to make the math concrete.","id":"im_fa55c42249498afd"}}
```

### Step 3: The GRPO Objective Function
Once we have calculated those relative Advantages ($A_i$), what do we do with them? We plug them right back into the exact same clipping mechanism we used in PPO! 

GRPO keeps PPO's brilliant solution to destructive updates. It still calculates the probability ratio $r_t(\theta)$ between the new and old policy, and it still clips that ratio.

The final GRPO objective for a group looks remarkably familiar:

$$J_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( r_i(\theta) A_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$

*(Note: The added $-\beta \mathbb{D}_{KL}$ at the end is a KL divergence penalty. Because we don't have a Critic, we add this penalty directly into the objective to ensure the model doesn't stray too far from its original reference behavior, $\pi_{ref}$, preventing it from "gaming" the reward system by outputting gibberish).*

### The Summary: Why GRPO wins for LLMs
1.  **Memory Efficiency:** By dropping the Critic network, you free up massive amounts of VRAM. You can train larger models on the same hardware.
2.  **Compute Efficiency:** No need to do forward or backward passes through a Critic.
3.  **Self-Balancing:** Because the Advantage is relative to a generated group, the algorithm naturally scales whether the overall task is very easy (all high rewards) or very hard (all low rewards). It always finds the *relatively* best actions to reinforce.

You've gone from the wild variance of REINFORCE, to the stable but memory-heavy Actor-Critic PPO, to the lean, mathematically elegant, group-normalized GRPO. 

Would you like to explore how they design the Reward Models (or Rule-Based Verifiers) that generate those raw scores in modern reasoning models?




In reinforcement learning for Large Language Models (LLMs), $\pi_{ref}$ stands for the **Reference Policy** (or Reference Model). 

Usually, it is an exact, frozen copy of the model from *before* the reinforcement learning phase began (often the Supervised Fine-Tuned or "SFT" model).

To understand why we need it, we have to look at a massive problem in RL called **Reward Hacking**.

### The Problem: Reward Hacking
Imagine you are training an LLM to write Python code, and your Reward Model gives points if the code successfully compiles and runs. 

Neural networks are notoriously lazy and literal. If the model figures out that printing `"Hello World"` technically compiles and runs perfectly every time, it might decide that the optimal strategy is to just output `"print('Hello World')"` no matter what prompt you give it. 

Alternatively, if you reward an LLM for being "polite," it might figure out that repeating the word "please" 500 times yields the highest possible politeness score, completely destroying its ability to speak normal English. The model "hacked" the reward, but ruined its general intelligence.

### The Solution: The Reference Model ($\pi_{ref}$)
To stop the model from forgetting how to speak normally while it chases high rewards, we introduce an anchor. This is where $\pi_{ref}$ comes in.

During training, we keep the frozen Reference Model ($\pi_{ref}$) sitting in GPU memory right next to the active model we are actually training ($\pi_\theta$). 

Every time our active model generates a token (a word) and calculates its probability, we ask the Reference Model: *"Hey, what probability would you have given that same word?"*

We compare the two using a mathematical formula called **KL Divergence** ($\mathbb{D}_{KL}$). 
* If the active model ($\pi_\theta$) and the reference model ($\pi_{ref}$) have similar probabilities, the KL Divergence is near zero. No penalty.
* If the active model starts changing its probabilities drastically to game the reward system, the KL Divergence shoots up.

### The Math: Adding the Penalty
This is why the GRPO (and PPO) objective function subtracts the KL Divergence from the Advantage:

$$ ... - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref})$$

Here, $\beta$ (beta) is just a tuning knob. If $\beta$ is high, the model is heavily penalized for deviating from the reference. 

**An Analogy:** Imagine you are a stand-up comedian ($\pi_\theta$) trying to get the most laughs (the Reward). However, your strict English teacher ($\pi_{ref}$) is in the front row. You are free to try new jokes to get more laughs, but if you start speaking pure gibberish just because it makes the crowd laugh, your teacher deducts points. You have to find a way to maximize laughs *while remaining somewhat close to proper English*.

That is the purpose of $\pi_{ref}$: it keeps the model grounded in reality while it explores how to maximize its reward.


















It is exactly the first one: **$\pi_{ref}$ is a frozen snapshot of the exact same model** from right before the reinforcement learning phase began. 

It is almost never an entirely different model, and it is architecturally completely different from a Critic. Here is exactly what it is and why it has to be that way.

### 1. The Timeline of the Snapshot
To train a modern LLM (like Llama 3 or DeepSeek), you go through distinct stages.
1.  **Pre-training:** Teach the model to predict the next word on massive amounts of internet text. (Result: Base Model)
2.  **Supervised Fine-Tuning (SFT):** Show the model high-quality examples of human conversations so it learns the format of questions and answers. (Result: SFT Model)
3.  **Reinforcement Learning (RL):** Use PPO or GRPO to optimize the model's reasoning or alignment.

Right before Step 3 begins, you take a copy of the **SFT Model**, freeze its weights so they can never change, and call it $\pi_{ref}$. The model you actually train and update during Step 3 is $\pi_\theta$.

### 2. Why it MUST be the same model
The math behind the KL Divergence penalty completely breaks down if $\pi_{ref}$ is a different model. 

When generating a response, for every single token, both the active model ($\pi_\theta$) and the reference model ($\pi_{ref}$) output a probability distribution over their entire vocabulary (e.g., 100,000 possible words). 

The KL Divergence formula directly compares these two probability distributions:
$$\mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) = \sum_x \pi_\theta(x) \log \left( \frac{\pi_\theta(x)}{\pi_{ref}(x)} \right)$$

* **Same Vocabulary:** If they were different models, they might have different tokenizers (meaning word #450 in one model is "apple" and in the other it's "car"). You cannot compare their probabilities.
* **Same Baseline Knowledge:** We are trying to measure how much the active model is *deviating* from its starting point. If you used a smaller or different model as a reference, the KL penalty would punish the active model simply for being smarter or differently trained than the reference, rather than punishing it for reward hacking.

### 3. How it differs from a Critic
The Critic in PPO and the Reference model in GRPO/PPO serve completely different purposes and output completely different shapes of data.

* **The Reference Model ($\pi_{ref}$):** Outputs a massive vector of probabilities (e.g., shape `[batch_size, sequence_length, vocab_size]`). Its job is purely to say, *"Here is what I would have said."*
* **The Critic ($V_\phi$):** Outputs a single scalar number (e.g., shape `[batch_size, sequence_length, 1]`). Its job is to say, *"I predict this state will result in a final score of 7.5."* ### The Ultimate Cost
Because $\pi_{ref}$ is an exact clone of your active model, if you are training a 70 Billion parameter model, $\pi_{ref}$ is *another* 70 Billion parameter model sitting in your GPU memory, doing nothing but inference. 

This is exactly why dropping the Critic in GRPO is so vital. If you used standard PPO on a 70B model, you would need to hold the Actor (70B), the Reference (70B), the Critic (usually another 70B), and the Reward Model (often another 70B). GRPO eliminates the Critic, saving massive amounts of VRAM while keeping the Reference model to prevent the actor from turning into a reward-hacking gibberish machine.











# Changes in reward functions 

- If all the test cases pass then give very high reward 
- If partial test cases pass then give positive small reward 
- If None of the tests pass then give negative reward 

Specifics 
-  >50% pass then +ve rewards 
- 0 - 50% -ve reward 
- If compilation fails then highly neg reward 
- If all test cases pass then very high positive rewards 



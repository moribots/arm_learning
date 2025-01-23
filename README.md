# Proximal Policy Optimization (PPO) for 7DOF Arm Control


PPO is a **policy gradient method** that directly optimizes the policy $\pi_\theta(a|s)$, aiming to maximize the expected cumulative reward. It introduces **clipping**, to ensure stable and efficient training.

---

## 2. Key Components of PPO

### (a) Policy and Value Networks
- **Policy Network** (Actor): Outputs actions for the robot to take (e.g torques).
- **Value Network** (Critic): Estimates the value of the current state $V(s)$, which is used to compute the **advantage function**.

### (b) Advantage Function
The advantage function measures how much better an action $a$ is compared to the average action:

$$
A(s, a) = R + \gamma V(s') - V(s)
$$

- $R$: Cumulative reward after taking action $a$.
- $\gamma$: Discount factor to weigh future rewards.
- $V(s)$: Estimated value of the current state.
- $V(s')$: Estimated value of the next state.

In the code:
- `rewards` store $R$.
- `values` store $V(s)$.

### (c) Clipped Surrogate Objective
PPO uses a clipped objective to limit the size of policy updates, ensuring stable training:

$$
L^{\text{CLIP}}(\theta) = \min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right)
$$

- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$: Ratio of new and old policy probabilities.
- $\epsilon$: Clipping parameter (`eps_clip` in the code).

In the code:

```
surr1 = ratio * advantage  
surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage  
policy_loss = -torch.min(surr1, surr2).mean()
```

### (d) Value Loss
PPO minimizes the error in state value estimation using the following loss:

$$
L^{\text{VALUE}} = \text{MSE}(V(s), R + \gamma V(s'))
$$

In the code:

```
value_loss = nn.MSELoss()(state_value, torch.tensor([advantage.item() + state_value.item()], dtype=torch.float32))
```

### (e) Reward Signal
The reward signal drives the robot to minimize the distance to a target position. In the code:

```
reward = -distance_to_target * 10  # Amplify reward signal
```

---

## 3. Pseudocode

### (a) Training Workflow
1. **Collect Experience**:
   - Simulate the robot and store:
     - States.
     - Actions taken by the policy.
     - Rewards.
     - State values (currently placeholders but can be replaced with critic outputs).
2. **Compute Advantages**:
   - Use rewards and value estimates to compute $A(s, a)$ for each step.
3. **Optimize Policy and Value Networks**:
   - Perform multiple gradient updates to improve:
     - **Policy Network**: Using the clipped surrogate loss.
     - **Value Network**: Using mean squared error loss.

### (b) Testing Workflow
1. **Load Trained Policy**:
   - Load the saved policy after training.
2. **Execute Actions**:
   - Use the trained policy to select joint torques.
   - The goal is to move the end-effector to predefined target positions.

---

## 4. PPO Mechanisms

1. **Multiple Updates per Batch**:
   - PPO optimizes the policy over several epochs (`k_epochs`) for better sample efficiency.

2. **Clipped Updates**:
   - Clipping ensures stable policy updates, reducing large jumps in policy probabilities.

3. **Reward Amplification**:
   - Rewards are amplified (`reward = -distance_to_target * 10`) to provide stronger learning signals.

---

## 5. Potential Improvements

1. **Critic Output**:
   Replace the placeholder values (`values.append(0.0)`) with actual predictions from the critic network.
2. **Reward Shaping**:
   Add intermediate rewards for approaching the target to accelerate learning.
3. **Normalize Inputs**:
   Normalize states and rewards to stabilize training and improve efficiency.

---

## 7. Summary


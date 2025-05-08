# Autonomous Driving Optimization using Reinforcement Learning

## Project Overview
This project implements reinforcement learning to develop autonomous driving agents that optimize vehicle routing and lane management while adhering to safety thresholds. We use Highway-Env's 2D simulation environment to train decision-making agents that can adapt to diverse traffic patterns and make intelligent decisions in complex driving scenarios.

## Team Members
- Shyam Akhil Nekkanti (8982123)
- Layanika V.S (8934459)
- Samyukth Lalith Lella Gopal (9005574)

## Problem Statement
Development of autonomous driving systems faces two critical challenges:
1. Limited adaptability to diverse traffic patterns
2. Poor decision-making in complex driving scenarios

Traditional rule-based systems cannot adapt to unexpected driving conditions, while supervised learning approaches require extensive labeled data and struggle with edge cases. Our project addresses these limitations by developing RL agents that learn optimal driving policies through interaction with Highway-Env's simulated driving scenarios.

## Environment
We use Highway-Env, a suite of environments for autonomous driving and tactical decision-making. Our implementation focuses on the highway-v0 environment, which simulates highway driving scenarios.

The environment includes:
- **State**: Position, speed, lane, and surrounding vehicle information
- **Actions**: Lane changes, acceleration/deceleration
- **Rewards**: Safety rewards (avoiding collisions), efficiency rewards (speed, staying in faster lanes)

## Algorithms
We are implementing and comparing three reinforcement learning algorithms:
- **DQN (Deep Q-Network)**: Deep learning model for Q-value approximation with experience replay and target networks
- **PPO (Proximal Policy Optimization)**: Policy gradient method with constrained updates
- **A2C (Advantage Actor-Critic)**: Combined value-based and policy-based approach

## Implementation Details

### Custom Reward Function
Our reward function balances safety and efficiency using the formula:
```
R(st, at, st+1) = w1 * (-α * max(0, (dthreshold - dcollision)/dthreshold)²) + w2 * (β1 * vt + β2 * Ifast_lane - β3 * Ilane_change)
```
Where:
- w1, w2: Weights for safety vs. efficiency (0.65, 0.35)
- α: Safety scaling factor (8.0)
- dthreshold: Safety distance threshold (0.8)
- β1, β2, β3: Efficiency parameters for velocity, lane preference, and lane changes
- Ifast_lane: Indicator for being in fast lane
- Ilane_change: Indicator for changing lanes

### DQN Architecture
Our Deep Q-Network implementation includes:
- Enhanced neural network for Q-value function approximation
- Experience replay buffer to improve sample efficiency
- Target network for stable learning
- Epsilon-greedy exploration strategy

Network Structure:
1. Input Layer: Flattened observation space
2. Hidden Layer 1: 512 neurons with ReLU activation
3. Hidden Layer 2: 256 neurons with ReLU activation
4. Hidden Layer 3: 128 neurons with ReLU activation
5. Output Layer: Action space size

### Training Process
The training process includes:
- Experience collection and storage in replay buffer
- Regular updates to DQN using minibatch gradient descent
- Periodic target network updates
- Evaluation to track performance improvements

## Progress and Results
We have successfully implemented and trained the DQN algorithm with the following observations:
- The agent learns to navigate highway traffic while maintaining safety
- Training reveals that the agent consistently prefers higher speed when safe
- Reward function successfully balances safety and efficiency

## Future Work
- Implement PPO and A2C algorithms for comparison
- Test agents in more complex scenarios (merge-v0, intersection-v0)
- Develop custom reward functions for different driving scenarios
- Create a hyperparameter optimization framework
- Build visualization tools to analyze agent behavior

## Success Metrics
We evaluate our RL agents based on:
1. Safety performance (collision avoidance rate)
2. Efficiency metrics (travel time, speed maintenance)
3. Scenario completion rate at different traffic densities
4. Training convergence speed

## Expected Outcomes
1. Comprehensive quantitative comparison of DQN, PPO, and A2C across all three Highway-Env scenarios
2. Optimized RL agents capable of safe and efficient autonomous driving with at least 90% collision avoidance
3. Analysis of the effect of different reward functions on driving behavior
4. Identification of the most suitable algorithm for each driving scenario

## Installation and Setup
```bash
# Install required packages
pip install highway_env gymnasium numpy matplotlib tqdm seaborn pandas torch
```

## Usage
```python
# Import necessary libraries
import gymnasium as gym
import highway_env
import numpy as np
import torch
# More imports...

# Create and configure environment
env = gym.make("highway-v0", render_mode="rgb_array")
# Configure environment parameters...

# Initialize DQN agent
agent = DQNAgent(observation_space, action_space)

# Train the agent
# ...

# Evaluate the agent
# ...
```

## References
- Highway-Env: https://github.com/Farama-Foundation/highway-env
- DQN: https://www.nature.com/articles/nature14236
- PPO: https://arxiv.org/abs/1707.06347
- A2C: https://arxiv.org/abs/1602.01783

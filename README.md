# Quantum Key Distribution (QKD) resource management optimization using RL

QKD networks are crucial to ensure secure communication, while effective resource management is a leading challenge. This work provides cutting-edge simulation and reinforcement learning technique with deep reinforcement learning for edge resource optimization in QKD networks. We present a novel environment retaining the delicacy of the task distribution over multiple edge nodes based on node capacity, task difficulty, and delay. Our solution employs a Double Deep Q-Network (DDQN) for maximizing the best policies of resource usage, which exhibit noteworthy load balance, latency optimization, and overall system efficiency. Our suggested DDQN-based scheduler achieves 820,523 bps secure key rate, a 15-30% improvement over traditional solutions, with a latency decrease of 25% for facilitation of encryption at increased speeds for live secure communication. The solution is also less costly, maximizing resource utilization and decreasing node overloading by 40%, while providing fair allocation and system reliability increase. The process further maximizes cost-efficiency with a minimized cost-per-secure-bit value of 0.01 - 0.058 USD over heuristic scheduling algorithms. 

DRL-based Scheduler**: Utilizes DDQN for intelligent decision-making in QKD edge networks.
-  **Secure Key Rate Optimization**: Achieves peak secure key rates up to **820,523 bps**.
-  **Latency Reduction**: Reduces delay by more than **25%** compared to baseline.
-  **Resource Load Balancing**: Prevents node overloading and improves fairness across nodes by over **40%**.
-  **Cost-Efficiency**: Optimizes cost-per-secure-bit as low as **$0.01**.


# Edge Resource Management in QKD Networks  
## An Advanced Simulation and Deep Reinforcement Learning Approach


---

## Project Overview

This project introduces a novel approach to optimize resource management in Quantum Key Distribution (QKD) networks using a Double Deep Q-Network (DDQN). A custom simulation environment models the interaction of quantum and classical components under realistic conditions, and a reinforcement learning agent is trained to dynamically allocate edge computing tasks in order to minimize latency, reduce node overload, and enhance secure key generation efficiency.

The proposed DDQN-based scheduling algorithm outperforms traditional heuristic and baseline RL methods in terms of secure key rate, latency, load balancing, and cost-efficiency.

---

## Key Contributions

- Development of a simulation environment that models BPSK-modulated quantum channels (with Rayleigh fading and AWGN noise).
- Implementation of an OpenAI Gym-based edge task allocation environment.
- Application of a DDQN-based reinforcement learning agent to optimize task scheduling across edge nodes.
- Visualization and evaluation of performance improvements in secure key rate, latency, cost per bit, and system reliability.

---

## Core Features

- **Quantum Channel Simulation:** Includes realistic physical impairments and noise modeling.
- **Edge Resource Environment:** Discrete action/state space with overload-aware reward structure.
- **DDQN Agent:** Stable learning via separate target and prediction networks, trained with experience replay and epsilon-greedy exploration.
- **Performance Metrics:** Secure key rate, latency, cost per secure bit, node load distribution, and cumulative rewards.
- **Validation:** Visualizations and performance tables compare DDQN to heuristic and standard DQN-based baselines.

---

## Methodology

### QKD Simulation

- **Modulation:** BPSK over Rayleigh + AWGN
- **Raw Key Rate:** 1 Mbps
- **Error Correction Efficiency:** 1.15
- **Privacy Amplification Factor:** 0.90

### Edge Resource Environment

- **State Space:** Normalized load of edge nodes
- **Action Space:** Discrete node selection
- **Reward:** Negative latency with penalties for overload

### DDQN Agent

- **Network Architecture:** Two hidden layers with 64 units each (ReLU)
- **Training Parameters:**
  - Discount factor γ = 0.99
  - Batch size = 64
  - Learning rate = 0.001
  - Target update frequency = every 10 episodes
  - Epsilon-greedy exploration with decay

---

## Results Summary

| Metric                          | DDQN-Based Approach      | Baseline DQN           | Heuristic Scheduling     |
|--------------------------------|--------------------------|------------------------|--------------------------|
| Secure Key Rate (bps)          | 820K – 140K              | 500K – 100K            | 700K – 200K              |
| Latency (sec)                  | 0.003 – 0.005            | 0.004 – 0.007          | 0.0035 – 0.006           |
| Cost per Secure Bit (USD)      | 0.01 – 0.058             | 0.004 – 0.007          | 0.015 – 0.05             |
| Load Balancing Efficiency      | High                     | Medium                 | Low                      |
| Overload Mitigation            | 40% improvement          | 20% improvement        | 5% improvement           |
| Latency Reduction              | 25%                      | 12%                    | 5%                       |
| Convergence Episodes           | ~4000                    | ~3000                  | N/A                      |

---

## Visualizations

- Secure Key Rate vs. Error Probability
- Latency vs. Error Rate
- Cumulative Reward over Episodes
- Load Distribution over Episodes
- Latency and Overload Comparison (DDQN vs. Baseline)

---

## How to Run

1. **Install Dependencies** (if not using Google Colab):
   ```bash
   pip install numpy pandas matplotlib gym

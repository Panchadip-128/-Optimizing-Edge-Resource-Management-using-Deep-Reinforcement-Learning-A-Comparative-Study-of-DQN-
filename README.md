# Quantum Key Distribution (QKD) resource management optimization using RL

QKD networks are crucial to ensure secure communication, while effective resource management is a leading challenge. This work provides cutting-edge simulation and reinforcement learning technique with deep reinforcement learning for edge resource optimization in QKD networks. We present a novel environment retaining the delicacy of the task distribution over multiple edge nodes based on node capacity, task difficulty, and delay. Our solution employs a Double Deep Q-Network (DDQN) for maximizing the best policies of resource usage, which exhibit noteworthy load balance, latency optimization, and overall system efficiency. Our suggested DDQN-based scheduler achieves 820,523 bps secure key rate, a 15-30% improvement over traditional solutions, with a latency decrease of 25% for facilitation of encryption at increased speeds for live secure communication. The solution is also less costly, maximizing resource utilization and decreasing node overloading by 40%, while providing fair allocation and system reliability increase. The process further maximizes cost-efficiency with a minimized cost-per-secure-bit value of 0.01 - 0.058 USD over heuristic scheduling algorithms. 

DRL-based Scheduler**: Utilizes DDQN for intelligent decision-making in QKD edge networks.
-  **Secure Key Rate Optimization**: Achieves peak secure key rates up to **820,523 bps**.
-  **Latency Reduction**: Reduces delay by more than **25%** compared to baseline.
-  **Resource Load Balancing**: Prevents node overloading and improves fairness across nodes by over **40%**.
-  **Cost-Efficiency**: Optimizes cost-per-secure-bit as low as **$0.01**.

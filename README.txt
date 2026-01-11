================================================================================
PROJECT: Neuro-Symbolic Autonomous Cyber Defense Agent using 
         Knowledge-Informed Learning (CAGE 4)
LAST UPDATED: 2026-01-11
================================================================================

1. OVERVIEW
This repository contains the official implementation of the paper "Neuro-Symbolic 
Autonomous Cyber Defense Agent using Knowledge-Informed Learning" published in 
[cite_start]IEEE Access[cite: 23, 25]. The framework integrates deep learning with symbolic 
reasoning through a 4-layer architecture (Network, Data, Decision, and Ontology 
[cite_start]Layers) to enhance cyber defense resilience[cite: 35, 36, 131].

2. CORE CONTRIBUTIONS
- Knowledge-based State Abstraction: Summarizes 11,000+ dimensional raw data 
  into dense knowledge vectors representing host infection and service status, 
  [cite_start]significantly reducing complexity[cite: 38, 65, 167].
- Priority-based Reward Shaping: Implements a potential-based reward mechanism 
  [cite_start]to prioritize the protection and recovery of critical network assets[cite: 38, 168].
- Mission-Aware Learning: Improves the system uptime rate and extends agent 
  [cite_start]survival time by leveraging domain knowledge within the reward signal[cite: 39, 40, 206].

3. EXPERIMENTAL ENVIRONMENT
- [cite_start]Simulator: CybORG CAGE Challenge 4 (DroneSwarm Scenario)[cite: 37, 182].
- [cite_start]RL Algorithm: Proximal Policy Optimization (PPO)[cite: 137, 173].
- [cite_start]Metrics: System Uptime Rate, Episode Length (Survival Time)[cite: 189, 206].

4. SYSTEM SPECIFICATIONS
- OS: Ubuntu 24.04 (via WSL2)
- [cite_start]Python Version: 3.10.14 [cite: 22]
- [cite_start]Key Dependencies: [cite: 22]
  * ray[rllib]==2.9.3
  * torch==2.2.2
  * gymnasium==1.2.2
  * CybORG==3.1

5. GETTING STARTED
# Step 1: Create Virtual Environment
python -m venv .venv
source .venv/bin/activate

# Step 2: Install Libraries
pip install ray==2.9.3 gymnasium==1.2.2 torch==2.2.2 pandas matplotlib

# Step 3: Install CybORG Simulator
cd third_party/CybORG
pip install -e .

# Step 4: Run Training
export NESY_LAM=1.0
python rllib_train_cyborg_with_metrics.py --exp-name "NeSy_Final"

6. REPOSITORY CONTENTS
- cyborg_env_maker.py: Environment wrapper implementing NeSy logic.
- rllib_train_cyborg_with_metrics.py: Main training script with PPO.
- [cite_start]requirements_actual.txt: Full list of tested library versions[cite: 22].
- README.txt: This instruction file.

================================================================================
CITATION: H.-G. Lee and K.-H. Lee, "Neuro-Symbolic Autonomous Cyber Defense 
Agent using Knowledge-Informed Learning," IEEE Access, 2026.
================================================================================
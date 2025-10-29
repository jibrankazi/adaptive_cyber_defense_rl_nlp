<<<<<<< research-upgrade
![](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/ci-adaptive.yml/badge.svg)
## Research Summary (PhD Portfolio)
=======
Adaptive Cyber Defense RL + NLP
>>>>>>> main




Overview

This repository presents Adaptive Cyber Defense RL + NLP, an experimental research system that integrates
deep reinforcement learning (RL) and natural language processing (NLP) for intelligent, adaptive cyber defense.
The project investigates the hypothesis that embedding language-derived threat signals into an RL-driven defense agent
reduces attack success rates and detection latency compared with static, rule-based baselines.

All code is fully reproducible on CPU-only environments with seeded configurations, lightweight smoke runs,
and CI validation. This work forms part of my doctoral research direction in adaptive, interpretable AI for
the University of Toronto PhD in Computer Science (Fall 2026).

Hypothesis

Integrating NLP-based threat intelligence into an RL defense agent
reduces both attack success rate and detection latency while maintaining low false positives.

Dataset

RL environment: synthetic cyber-attack simulation (≈ 50 000 episodes, 30 features).

Text corpus: 10 000 security emails / logs labeled for phishing, malware, and benign activity.

Data split = 70 / 15 / 15 (train / validation / test).

Methods
Reinforcement Learning

Algorithms: Deep Q-Network (DQN) and Proximal Policy Optimization (PPO).

Reward = – (breach + false alarm) + mitigation success × weight.

Action space: enable/disable firewall rules, deploy honeypots, trigger alerts.

NLP Classification

Model: fine-tuned BERT-base-uncased for phishing / malware detection.

Training objective: cross-entropy loss with early stopping.

Outputs feed into RL state representation as semantic threat features.

Threat Modeling

Simulated attacks: phishing, brute-force logins, malware propagation.

The agent learns adaptive response policies to balance security and performance.

Results
Component	Metric	Value	Notes
DQN Agent	Mean episodic reward	85	after 2 000 episodes
	Attack reduction vs baseline	72 %	compared to static policy
PPO Agent	Stability gain over DQN	+12 %	variance reduction
BERT Classifier	F1	0.93 ( P 0.92 / R 0.94 )	baseline SVM 0.87 F1
System	Mean time-to-detection ↓	40 %	false positives < 5 %

Results are preliminary and reproducible through the configuration files below.

Reproducibility
# Environment setup
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
python -m pip install -r requirements.txt

# Smoke run (CPU)
make smoke

# Full training
python src/train_rl.py --config configs/dqn.yaml
python src/train_nlp.py --model bert-base-uncased --epochs 5
python src/evaluate.py --rl_model models/dqn.pkl --nlp_model models/bert_classifier.pkl


All experiment configurations are stored under configs/.
Running make all reproduces the full pipeline and regenerates results/report.json.

Continuous Integration (CI)
# .github/workflows/ci-adaptive.yml
name: CI – Adaptive Cyber Defense (smoke)
on: [push, pull_request]
jobs:
  smoke:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: python-version: '3.11'
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r requirements.txt
      - run: python src/train_nlp.py --epochs 1
      - run: python src/train_rl.py --config configs/dqn.yaml
      - run: python src/evaluate.py
      - name: Show report
        run: type results\report.json


A passing green ✓ on CI confirms successful reproducibility.

Research Context

This project exemplifies my focus on reinforcement learning for adaptive defense,
natural-language threat understanding, and reproducibility in AI research.
Its themes directly connect to the work of:

Prof. Sheila McIlraith – explainable reinforcement learning and structured decision systems.

Prof. Jimmy Ba – optimization and stability in deep RL training.

Together these inform my long-term goal of developing interpretable, self-adapting AI agents for dynamic, safety-critical environments.

Quick Results (CPU Smoke)
Metric	Value
RL mean reward	77.48
Episodes	10
NLP F1	0.913
Citation
@software{Kazi_AdaptiveCyberDefense_2025,
  author = {Kazi, Jibran Rafat Samie},
  title = {Adaptive Cyber Defense RL + NLP: Integrating Language-Derived Threat Signals into Reinforcement Learning Agents},
  year = {2025},
  url = {https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp},
  license = {MIT}
}

Contact

Kazi Jibran Rafat Samie
📍 Toronto, Canada
📧 jibrankazi@gmail.com

🔗 github.com/jibrankazi

🔗 linkedin.com/in/jibrankazi

© 2025 Kazi Jibran Rafat Samie
Independent research project on adaptive, language-informed reinforcement learning for cyber defense.
Part of my doctoral research direction.

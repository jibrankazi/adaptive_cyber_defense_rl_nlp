# Adaptive Cyber Defense RL + NLP — verified actual-data research pipelines

This project now has **two complete, separately tested public-data machine-learning pipelines**: historic phishing-website feature classification and UCI SMS spam review recommendations. Neither pipeline fabricates source observations. **This is not a deployed cyber defense platform or a validated reinforcement-learning agent.**

## 1. Genuine UCI phishing website classifier

[Successful independent source-to-model GitHub Actions run](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/runs/38006875632)

The original [UCI Phishing Websites dataset 327](https://archive.ics.uci.edu/dataset/327/phishing+websites) has **11,055 labeled historical website records**, including **4,898 phishing** and **6,157 legitimate** cases with **30 already-extracted numeric attributes**. A GroupShuffleSplit keeps identical attribute patterns out of different partitions. Model selection is based on validation data only. The **2,296-row independent test partition** produced these measured results:

| Original 2012 website holdout | Measured |
| --- | ---: |
| Random Forest accuracy | **95.60%** |
| Phishing precision | **94.22%** |
| Phishing recall | **95.48%** |
| Phishing F1 | **94.85%** |
| ROC AUC | **0.9914** |
| True phishing sites missed | **44** |
| Legitimate sites wrongly flagged | **57** |

A baseline and logistic regression were evaluated against the same test records, and all outputs were independently recalculated in GitHub Actions. **These scores do not establish protection against current phishing campaigns**. The data do not include raw website URLs, and there is no live URL-to-attributes extraction or browser blocking.

Run locally (after installing requirements):

    python -m src.real_phishing_websites

The program saves the true original-data test scores, trained selected model, publisher information, and exact feature names to results/real_uci_phishing/. The optional --predict-features-json option accepts only a JSON object containing all thirty correctly encoded, already extracted UCI features. It does not visit or fetch websites.

## 2. Genuine UCI SMS NLP screening and recommendation model

[Successful independent complete public-SMS-source-to-inference run](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/runs/38006990912)

The [UCI SMS Spam Collection dataset 228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) contains **5,574 actual published labeled SMS messages**. Removing **403 duplicate texts** before splitting gives **5,171 unique messages**, with 3,102 training, 1,034 validation and 1,035 held-out records. A trained TF-IDF/logistic model had **0.9587 average precision** and **0.9863 ROC AUC** on the real heldout SMS records.

The validation-only selection of review threshold **0.36** obtained real test-spam recall **92.37%**, while a fixed 0.50 threshold obtained **90.08%** recall. However, the policy's manually assumed review/miss cost on the untouched test records was **168** at the tuned threshold versus **154** at the fixed threshold: tuning did **not** improve the test-set example cost. These units are **artificial decision preferences, not observed financial, security or incident outcomes**.

Run the real historical SMS research workflow and local inference:

    python -m src.real_sms_defense
    python -m src.real_sms_defense --predict-text "Your appointment is tomorrow."
    python scripts/generate_plot.py

The CLI returns a score and a proposed review/allow label only. It cannot intercept, block, delete or quarantine a message. Raw original SMS message bodies are not included in the output artifacts.

## Setup and independent verification

    python -m pip install -r requirements.txt
    make test
    make smoke
    make phishing
    make all

GitHub Actions workflows independently execute and audit the genuine-source models and save source-to-heldout evidence:

- [Verified historical phishing website model, 11,055 source records](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/real-phishing-websites.yml)
- [Verified historical SMS NLP review recommendation model, 5,574 published message rows](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/real-sms-defense.yml)
- [Earlier original UCI SMS baseline](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/ci-adaptive.yml)

The old script that **invented a 72% reduction in successful attacks** has been replaced with a plot built from actual heldout SMS label counts and **clearly hypothetical** cost preferences. Its old unsupported attack-reduction image was removed.

## Important limits

The repository's historical research claims about trained DQN/PPO, a fine-tuned BERT threat detector, RL-based 72% attack reduction, phishing/malware defense, and real response-latency improvement are **not proven** by the existing original-source evidence. The repository's small random cyber environment remains a toy simulator. The offline RL entry point deliberately refuses to invent a policy or rewards without authorized source logs of real defense state/action/response/outcomes and a valid counterfactual evaluation design.

The two UCI pipelines are successful supervised learning demonstrations on **historic publicly collected data**, not current live cyber defense or safe autonomous blocking. They make no claim of real-world robustness, recent fraud prevention, or financial return. Loading local joblib model files is only appropriate when the saved artifacts are your own or fully trusted.

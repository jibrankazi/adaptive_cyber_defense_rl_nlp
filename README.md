# Adaptive Cyber Defense RL + NLP — Verified real-data components

**Two real-source classification workflows now run end to end in GitHub Actions:** (1) historical phishing website risk detection on actual UCI labeled site feature vectors, and (2) actual SMS spam text classification with a local review recommendation. Both have trained classifiers, disjoint evaluation examples, true labels, saved model files, and independently checked result artifacts.

**Not implemented or verified:** real network intrusion prevention, live website scanning, BERT fine-tuning, DQN/PPO training, real security interventions, or operational attack-reduction claims. The former 72% attack-reduction chart was fabricated and was removed; the chart generation script now uses only actual UCI SMS holdout labels and explicitly hypothetical scenario costs.

## Working project A: Phishing website classification

[Successful genuine 11,055-row UCI site classification run](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/runs/38006758425).

Original source: [UCI Phishing Websites #327](https://archive.ics.uci.edu/dataset/327/phishing+websites), Mohammad & McCluskey, DOI [10.24432/C51W2X](https://doi.org/10.24432/C51W2X), licensed CC BY 4.0. The original records are actual historically annotated phishing and legitimate sites from PhishTank, MillerSmiles and web search. The source contains 30 already-extracted website features **but no raw URLs**.

| Actual evidence | Value |
| --- | ---: |
| Original labeled website records | **11,055** |
| Phishing / legitimate original records | **4,898 / 6,157** |
| Training / validation / untouched test records | 6,527 / 2,232 / **2,296** |
| Exact duplicate 30-feature patterns crossing splits | **0** |
| Model selected using validation set only | Random Forest |
| Heldout phishing accuracy | **95.60%** |
| Heldout phishing precision | **94.22%** |
| Heldout phishing recall | **95.48%** |
| Heldout phishing F1 | **94.85%** |
| Heldout ROC AUC | **0.9914** |
| Phishing missed / legitimate wrongly flagged | **44 / 57** |

The test rows' original numeric features and labels were not used to choose the model. This is a feature-pattern-disjoint **random** historical test, not a time-forward study. It does not establish detection accuracy on current phishing campaigns, online URL processing or any automatic site blocking. Some of the 30 features depend on page content, reputation and domain/index data that may be unavailable when first encountering a site.

Run:

~~~
python -m pip install -r requirements.txt
python -m src.real_phishing_websites
~~~

Results: results/real_uci_phishing/verified_results.json, observed_uci_website_heldout_scores.csv, selected_uci_phishing_model.joblib and ordered_original_feature_columns.json.

To score one **already extracted** complete 30-field website feature JSON using the saved model, run:

~~~
python -m src.real_phishing_websites --predict-features-json YOUR_30_FIELDS.json
~~~

The input keys must match the original feature names in ordered_original_feature_columns.json, all values must be -1/0/1. **This is not a raw URL scanner.**

## Working project B: Actual UCI SMS spam message NLP and human review triage

[Successful genuine SMS source-to-heldout classifier and local inference run](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/runs/38006589097).

Original source: [UCI SMS Spam Collection #228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), Almeida & Hidalgo 2011, DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84), CC BY 4.0.

| Actual evidence | Value |
| --- | ---: |
| Original published SMS messages | **5,574** |
| Duplicate SMS texts excluded before splitting | **403** |
| Unique original messages | **5,171** |
| Train / validation / untouched test messages | 3,102 / 1,034 / **1,035** |
| Actual original test spam labels | 131 |
| Heldout spam average precision | **0.9587** |
| Heldout spam ROC AUC | **0.9863** |
| Fixed threshold 0.50: heldout spam F1 | **0.9042** |
| Validation-selected threshold 0.36: heldout spam F1 | **0.8462** |

The program trains word/bigram TF-IDF and a balanced logistic-regression spam classifier on actual message text. It chooses the optional review probability threshold from **validation only**, using explicitly **hypothetical** costs of 2 points per mistaken ham review and 10 points per missed spam. These are **not real money, measured policy rewards or actual user interactions**.

**An important negative result:** On 1,035 completely separate test messages, the validation-selected 0.36 threshold had **168 hypothetical mistake-cost points**, whereas a fixed 0.50 threshold had **154**. The tuned policy did not improve that independent test outcome; the true result is reported rather than hidden.

Run the actual pipeline and score a message locally (no paid API key needed):

~~~
python -m src.real_sms_defense
python -m src.real_sms_defense --predict-text "Reminder: your appointment is tomorrow."
python scripts/generate_plot.py
~~~

Outputs: results/real_uci_sms_triage/verified_results.json, uci_sms_review_model.joblib, review_threshold.json, heldout_observed_sms_scores.csv (does not include raw public SMS bodies), heldout_precision_recall.png and heldout_assumed_review_costs.png. The local CLI recommends "review" or "allow" but does not intercept, quarantine, delete or block messages.

## Reproduce and verify

~~~
python -m pip install -r requirements.txt
make test
make smoke
make phishing
make all
~~~

The Makefile was fixed: its prior smoke commands called nonexistent flags and a nonexistent experiment, whereas the current commands run the actual source-backed code. The software tests use small **fixtures only**; the empirical measurements above come from actual downloaded UCI publisher datasets.

Source-verified GitHub Actions:
- [Original real phishing website classifier and heldout source audit](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/real-phishing-websites.yml)
- [Original real SMS spam classifier, review decision and local prediction check](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/real-sms-defense.yml)
- [Earlier standalone real UCI SMS model baseline](https://github.com/jibrankazi/adaptive_cyber_defense_rl_nlp/actions/workflows/ci-adaptive.yml)

**Security:** Only load joblib model files you trained or trust, because they deserialize Python objects.

## Unimplemented reinforcement learning

The old DQN/PPO/BERT claims, 72% attack reduction and 40% detection latency improvement lack observed supporting experiments. The RL training entry point refuses to fabricate a model without authorized true security state/action/outcome logs and proper offline evaluation. The old gym environment contains random toy dynamics; it is not a real defense system or an experiment on true intrusion logs.

Working today: **two real-data historical supervised cyber/message classification pipelines with reproducible training, heldout evaluation and saved artifacts**. Not working: operational adaptive security RL, live interventions, or claims of current real-world accuracy.

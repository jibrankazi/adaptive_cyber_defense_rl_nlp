import json, os
os.makedirs("results", exist_ok=True)

def read_json(p, default):
    try: return json.load(open(p))
    except Exception: return default

rl  = read_json("models/rl_metrics.json", {"mean_reward": None, "episodes": 0})
nlp = read_json("models/nlp_metrics.json", {"f1": None})
report = {"rl_mean_reward": rl["mean_reward"], "rl_episodes": rl["episodes"], "nlp_f1": nlp["f1"]}
json.dump(report, open("results/report.json","w"), indent=2)
print("Evaluation report written → results/report.json:", report)

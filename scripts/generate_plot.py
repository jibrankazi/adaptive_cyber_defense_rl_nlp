"""Plot *actual evaluated* UCI SMS label-derived review costs, not fake RL results.

The original script drew invented '72% attack reduction' curves with random
episode numbers. It must never be mistaken for evaluated cyber attack data.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_actual_scenario_costs(metrics, destination):
    payload = json.loads(Path(metrics).read_text(encoding="utf-8"))
    if payload.get("data_source") != "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip":
        raise ValueError("Only actual original UCI SMS review reports supported")
    scenarios = payload.get("untouched_test", {}).get("policy_comparisons", {})
    keys = [
        "allow_all", "review_all",
        "fixed_probability_0_5", "selected_on_validation_only",
    ]
    if set(scenarios) != set(keys):
        raise ValueError("Expected four empirically evaluated comparison decisions")
    names = [
        "Allow every message",
        "Review every message",
        "Model, threshold = 0.50",
        "Model, threshold tuned on validation",
    ]
    values = [scenarios[key]["assumed_cost_units"] for key in keys]
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Expected actual heldout-label-derived nonnegative cost units")
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values)
    ax.bar_label(bars, padding=4)
    ax.set(
        title="UCI historical SMS test labels: ASSUMED triage-review cost units",
        ylabel="Assumed total error cost (lower is better; NOT real money)",
    )
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"Saved real-label-derived scenario comparison to {target}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", default="results/real_uci_sms_triage/verified_results.json")
    p.add_argument("--output", default="results/real_uci_sms_triage/heldout_assumed_review_costs.png")
    args = p.parse_args()
    plot_actual_scenario_costs(args.metrics, args.output)

"""Read genuine NLP evaluation metrics; do not invent RL performance."""
import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="models/nlp_metrics.json")
    args = p.parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if "source" not in metrics or "heldout_spam_f1" not in metrics:
        raise ValueError("Expected provenance-backed real NLP evaluation report")
    print(json.dumps({
        "source": metrics["source"],
        "task": metrics["task"],
        "heldout_spam_f1": metrics["heldout_spam_f1"],
        "rl_result": "UNAVAILABLE - real defense trajectories and model not implemented",
    }, indent=2))


if __name__ == "__main__":
    main()

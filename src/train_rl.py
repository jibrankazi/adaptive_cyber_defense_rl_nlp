"""RL training is deliberately disabled until real logged defense trajectories exist.

No synthetic attack episodes or placeholder reward values are generated.
Training requires action, state, outcome and intervention cost trajectories
plus a separately reviewed offline RL evaluation protocol.
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", required=True, help="Real, authorized, anonymized defense decision logs")
    args = p.parse_args()
    raise NotImplementedError(
        "Offline RL on documented real defense trajectories is NOT implemented. "
        "No model or performance metric has been generated."
    )


if __name__ == "__main__":
    main()

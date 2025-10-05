.PHONY: all train eval paper smoke

smoke:  ## quick CPU smoke
	python -m pip install -r requirements.txt
	python src/train_nlp.py --epochs 1
	python src/train_rl.py --config configs/dqn.yaml
	python src/evaluate.py

all: smoke
	@echo "Add paper build here later (LaTeX)."

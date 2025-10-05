.PHONY: smoke baselines plots paper test
smoke:
	python -m pip install -r requirements.txt
	python src/train_nlp.py --epochs 1
	python src/train_rl.py --config configs/dqn.yaml
	python src/evaluate.py

baselines:
	python src/run_experiment.py --config configs/ablations.yaml

plots:
	python src/plots.py  # or individual calls within your eval

paper:
	@echo "Compile paper/main.tex here (pdflatex) or build Markdown → PDF"
	# pdflatex -output-directory=paper paper/main.tex

test:
	pytest -q

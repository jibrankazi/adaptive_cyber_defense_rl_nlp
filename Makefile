.PHONY: setup test smoke sms phishing all plots help

help:
	@echo "setup     Install genuine-data pipeline requirements"
	@echo "test      Run software-contract tests (fixtures are NOT research data)"
	@echo "sms       Train/evaluate real UCI SMS NLP and saved local review model"
	@echo "phishing  Train/evaluate real UCI phishing website risk classifiers"
	@echo "plots     Visualize observed UCI SMS heldout review policy assumptions"
	@echo "smoke     Run tests then genuine SMS lifecycle and source-backed report"
	@echo "all       Full observed-source SMS + phishing + charts lifecycle"

setup:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q tests

sms:
	python -m src.real_sms_defense

phishing:
	python -m src.real_phishing_websites

plots:
	python scripts/generate_plot.py

smoke: test sms plots

all: test sms phishing plots

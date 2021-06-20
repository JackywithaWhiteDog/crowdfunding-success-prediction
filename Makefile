.PHONY: help clean
.DEFAULT_GOAL := help

datasetURL = https://drive.google.com/uc?id=17CYSbzt3rqCWcyJ5XeTipMBz_mlCM4M2

help:
	@echo -e "Usage: make [target] ...\n"
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

clean: ## Clean downloaded data.
	rm -f data/raw/*

install: ## Install dependencies
	python -m pip install -qU setuptools wheel
	python -m pip install -qr requirements.txt

data/raw/projects.csv:
	python src/data/download.py --url $(datasetURL) --output data/raw/projects.csv

data/interim/tokens.pickle: data/raw/projects.csv
	python src/preprocessing/get_tokens.py \
	--resource data/raw/projects.csv \
	--output data/interim/tokens.pickle

data/interim/token_stem.pickle data/interim/doc_stem.pickle: data/interim/tokens.pickle
	python src/preprocessing/text.py \
	--resource data/interim/tokens.pickle \
	--token data/interim/token_stem.pickle \
	--doc data/interim/doc_stem.pickle

.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y twelo || :
	@pip install -e .


install:
	@pip install --upgrade pip
	@pip install -e .
	@if [ ! -d "${PWD}/data" ]; then \
		mkdir ${PWD}/data; \
	fi


setup_virtual_env:
	@if pyenv virtualenvs | grep -q "twelo"; then \
		echo "Virtual environment 'twelo' already exists."; \
	else \
		pyenv virtualenv 3.10.6 twelo; \
	fi
	@pyenv local twelo

notebook_extensions:
	@jupyter contrib nbextension install --user
	@jupyter nbextension enable toc2/main
	@jupyter nbextension enable collapsible_headings/main
	@jupyter nbextension enable spellchecker/main
	@jupyter nbextension enable code_prettify/code_prettify

reset_local_data:
	@rm -rf data/*
	@mkdir data/twelo/
	@mkdir data/twelo/comp
	@mkdir data/twelo/raw
	@mkdir data/twelo/train
	@mkdir data/twelo/test
	@mkdir data/input
	@mkdir data/output
	@if [ ! -d "models" ]; then \
		mkdir models; \
	fi


download_train_data:
	@if [ ! -d "${PWD}/data/twelo/train" ]; then \
		mkdir ${PWD}/data/twelo/train; \
	fi
	@gcloud storage cp -r gs://kestrix/data/train ${PWD}/data/twelo
	@if [ ! -d "${PWD}/data/twelo/raw" ]; then \
		mkdir ${PWD}/data/twelo/raw; \
	fi

download_test_data:
	@if [ ! -d "${PWD}/data/twelo/test" ]; then \
		mkdir ${PWD}/data/twelo/test; \
	fi
	@gcloud storage cp -r gs://kestrix/data/test ${PWD}/data/twelo

download_models:
	@gcloud storage cp -r gs://kestrix/models .

upload_models:
	@gcloud storage cp -r models gs://kestrix/models

run_api:
	uvicorn twelo.api.fast:app --reload

run_train:
	@python -c "from twelo.model import train_model; train_model()"
	@gcloud storage cp -r models gs://kestrix/models .

run_test:
	@ls models/
	@read -p 'Select model without file ending: ' model_name && \
	python -c "from twelo.model import test_model; test_model('$$model_name')"

run_test_all:
	@python -c "from twelo.model import test_all_models; test_all_models()"

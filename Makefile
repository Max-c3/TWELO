.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y kestrix || :
	@pip install -e .


install:
	@pip install --upgrade pip
	@pip install -e .
	@if [ ! -d "${PWD}/data" ]; then \
		mkdir ${PWD}/data; \
	fi


setup_virtual_env:
	@if pyenv virtualenvs | grep -q "kestrix"; then \
		echo "Virtual environment 'kestrix' already exists."; \
	else \
		pyenv virtualenv 3.10.6 kestrix; \
	fi
	@pyenv local kestrix

notebook_extensions:
	@jupyter contrib nbextension install --user
	@jupyter nbextension enable toc2/main
	@jupyter nbextension enable collapsible_headings/main
	@jupyter nbextension enable spellchecker/main
	@jupyter nbextension enable code_prettify/code_prettify

reset_local_data:
	@rm -rf data/*
	@mkdir data/kestrix/
	@mkdir data/kestrix/comp
	@mkdir data/kestrix/raw
	@mkdir data/input
	@mkdir data/output
	@mkdir models


download_train_data:
	@if [ ! -d "${PWD}/data/kestrix/comp" ]; then \
		mkdir ${PWD}/data/kestrix/comp; \
	fi
	@gcloud storage cp -r gs://kestrix/data/comp ${PWD}/data/kestrix
	@if [ ! -d "${PWD}/data/kestrix/raw" ]; then \
		mkdir ${PWD}/data/kestrix/raw; \
	fi
	@gcloud storage cp -r gs://kestrix/data/raw ${PWD}/data/kestrix

download_models:
	@gcloud storage cp -r gs://kestrix/models models

upload_models:
	@gcloud storage cp -r models gs://kestrix/models

run_train:
	python -c "from kestrix.model import train_model; train_model()" 2>&1 | tee -a logfile.log
	gcloud storage cp -r models gs://kestrix/models

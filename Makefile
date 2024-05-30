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
	@rm -rf/data/*

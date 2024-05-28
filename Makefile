.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y kestrix || :
	@pip install -e .

install:
	@pip install --upgrade pip
	@if [ ! -d "${PWD}/data" ]; then \
		mkdir ${PWD}/data; \
	fi
	@pip install -e .

setup_virtual_env:
	@if pyenv virtualenvs | grep -q "kestrix"; then \
		echo "Virtual environment 'kestrix' already exists."; \
	else \
		pyenv virtualenv 3.10.6 kestrix; \
	fi
	@pyenv local kestrix

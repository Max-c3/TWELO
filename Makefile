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
	@pyenv virtualenv kestrix
	@pyenv local kestrix

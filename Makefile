.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
install_package:
	@pip install -e .

run_api:
	uvicorn mlb.api.fast:app --reload

clean:
    @rm -f */version.txt
    @rm -f .coverage
    @rm -f */.ipynb_checkpoints
    @rm -Rf build
    @rm -Rf */__pycache__
    @rm -Rf */*.pyc

all: install clean


SHELL := /bin/bash

STACK_NAME=dev6
BASE_STACK_NAME=ml-infra/dev
PULUMI_CMD=pulumi --non-interactive --cwd infra/


install-deps:
	curl -fsSL https://get.pulumi.com | sh
	npm install -C infra/

	sudo pip3 install poetry --upgrade
	poetry install

train:
	poetry run train


deploy: 
	$(PULUMI_CMD) stack init $(STACK_NAME) || true 
	$(PULUMI_CMD) config set aws:region us-west-2
	$(PULUMI_CMD) config set baseStackName $(BASE_STACK_NAME)
	$(PULUMI_CMD) config set runID $(shell poetry run train | awk '/Run ID/{print $$NF}')
	$(PULUMI_CMD) config 
	$(PULUMI_CMD) up --yes
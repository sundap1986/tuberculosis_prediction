.PHONY: install format train eval update-branch hf-login push-hub deploy all

install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black *.py 

train:
	python ./notebooks/notebook.py

eval:
	echo "## Model Metrics" > report.md
	cat ./reports/metrics.txt >> report.md
	
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./reports/confusion_matrix.png)' >> report.md
	
	cml comment create report.md

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add . 
	git commit -m "Update with new results"
	git push --force origin HEAD:update

hf-login: 
	pip install -U "huggingface_hub[cli]"
	git pull origin update
	git switch update
	huggingface-cli login --token "$(HF)" --add-to-git-credential

push-hub: 
	huggingface-cli upload sundararao/drug-classification ./ --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload sundararao/drug-classification ./models --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload sundararao/drug-classification ./reports --repo-type=space --commit-message="Sync Metrics"

deploy: hf-login push-hub

all: install format train eval update-branch deploy

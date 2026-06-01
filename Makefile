# Run with: make <target>
# On Windows without `make`, just copy the command after the tab.

PY = python

install:
	$(PY) -m pip install -r requirements.txt

hello:
	$(PY) hello.py

run:
	$(PY) -m agent.cli

eval:
	$(PY) -m eval.run --prompt-set prompts/eval.csv --system-prompt A

eval-promptB:
	$(PY) -m eval.run --prompt-set prompts/eval.csv --system-prompt B

test:
	pytest -q

.PHONY: install hello run eval eval-promptB test

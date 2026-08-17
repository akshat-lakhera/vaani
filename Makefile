PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv deps test inspect ingest bench ablate serve

venv:
	/opt/homebrew/bin/python3.11 -m venv .venv
	$(PIP) install --upgrade pip

deps:
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

inspect:
	$(PY) scripts/inspect_dataset.py --langs hi,mr --split validation --limit 400

ingest:
	$(PY) scripts/ingest.py --strategy whole --max-passages 25000 --eval-queries 400

bench:
	$(PY) scripts/bench.py --n 200

ablate:
	$(PY) scripts/ablate.py --n 120

e2e:
	$(PY) scripts/e2e_voice.py

serve:
	TOKENIZERS_PARALLELISM=false $(PY) -m vaani.api

IMAGE ?= mc-recon
TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
PORT  ?= 8000

.PHONY: install test eval mock lint lock image run smoke

install:            ## dev install
	pip install -e ".[dev]"

test:               ## unit + flow + API tests (no network, no LLM)
	pytest

eval:               ## every harness scenario, as a scorecard (no network, no LLM)
	mc-recon eval

mock:               ## serve the mock Disseqt and LLM endpoints on :8100
	mc-recon mock

lint:
	ruff check src tests scripts && ruff format --check src tests scripts

lock:               ## regenerate hash-pinned lock files after changing dependencies
	uv pip compile pyproject.toml --extra agent --universal --python-version 3.12 \
	  --generate-hashes -q -o requirements/runtime.lock.txt --custom-compile-command "make lock"
	uv pip compile requirements/build.in --universal --python-version 3.12 \
	  --generate-hashes -q -o requirements/build.lock.txt --custom-compile-command "make lock"

image:              ## build the container image
	docker build --build-arg REVISION=$(TAG) -t $(IMAGE):$(TAG) -t $(IMAGE):latest .

run:                ## run the image locally with a persistent volume
	docker run --rm -p $(PORT):8000 -v mc-recon-data:/app/var --read-only --tmpfs /tmp \
	  --cap-drop ALL --security-opt no-new-privileges $(IMAGE):latest

smoke:              ## black-box test a running instance
	python scripts/smoke_test.py http://localhost:$(PORT)

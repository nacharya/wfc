DOCKER      = docker
IMG_NAME    = wfc
CTR_NAME    = wfc
IMG_VERSION = latest
IMG_TAG     = $(IMG_NAME):$(IMG_VERSION)
PORT        = 9900:9900
VOLDATA     = $(shell pwd)/data

# ----------------------------------------------------------------- venv paths

VENV        = wfcvenv
VENV_BIN    = $(VENV)/bin
VENV_PY     = $(VENV_BIN)/python3
VENV_PIP    = $(VENV_BIN)/pip
VENV_ST     = $(VENV_BIN)/streamlit
VENV_JUP    = $(VENV_BIN)/jupyter

# ----------------------------------------------------------------- local dev

## Create (or recreate) the wfcvenv virtual environment and install requirements
venv:
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "Note: active venv $$VIRTUAL_ENV will be bypassed — $(VENV) binaries used directly."; \
	fi
	python3 -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip --quiet
	$(VENV_PIP) install -r requirements.txt
	@echo ""
	@echo "  wfcvenv ready."
	@echo "  To activate manually : source $(VENV)/bin/activate"
	@echo "  To run the web UI    : make web"
	@echo "  To run the notebook  : make notebook"

## Run the Streamlit web UI using wfcvenv (run 'make venv' first)
web:
	@if [ ! -f "$(VENV_ST)" ]; then \
		echo "wfcvenv not found — run: make venv"; exit 1; \
	fi
	cd app && ../$(VENV_ST) run wfc_main.py --server.port 9900

## Open the Jupyter notebook using wfcvenv (run 'make venv' first)
notebook:
	@if [ ! -f "$(VENV_JUP)" ]; then \
		echo "wfcvenv not found — run: make venv"; exit 1; \
	fi
	$(VENV_JUP) notebook app/wfc_doc.ipynb

## Show CLI usage help
cli:
	@echo ""
	@echo "  cd app && ./wfc --help"
	@echo ""
	@echo "  Common commands:"
	@echo "    ./wfc list"
	@echo "    ./wfc ticker <portfolio> [list|add <TK>|del <TK>]"
	@echo "    ./wfc gather  <portfolio> <start_date> <end_date>"
	@echo "    ./wfc predict <ticker>    <YYYY-MM-DD | days>"
	@echo "    ./wfc news    [<ticker> | owned]"
	@echo "    ./wfc oa      [<days>]"
	@echo ""

# ----------------------------------------------------------------- Docker

## Build the Docker image
$(IMG_NAME):
	$(DOCKER) build -t $(IMG_TAG) --no-cache -f Dockerfile .

all: $(IMG_NAME)

## Run the Docker container (maps port 9900, mounts ./data volume)
run:
	mkdir -p data
	$(DOCKER) run -tid -p $(PORT) -v $(VOLDATA):/data --name $(CTR_NAME) $(IMG_NAME)

shell:
ifeq ($(OS),Windows_NT)
	winpty $(DOCKER) exec -ti $(CTR_NAME) /bin/bash
else
	$(DOCKER) exec -ti $(CTR_NAME) /bin/bash
endif

save:
	$(DOCKER) commit $(CTR_NAME) $(IMG_TAG)

stop:
	$(DOCKER) stop $(CTR_NAME)
	$(DOCKER) rm   $(CTR_NAME)

logs:
	$(DOCKER) logs $(CTR_NAME)

show:
	$(DOCKER) ps -a | grep $(CTR_NAME)

rmi:
	$(DOCKER) rmi $(IMG_TAG)

prune:
	$(DOCKER) system prune -f

clean:
	rm -rf data
	rm -rf wfcvenv

.PHONY: all venv web notebook cli run shell save stop logs show rmi prune clean

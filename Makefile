DOCKER=docker

IMG_NAME=wfc
CTR_NAME=wfc
IMG_VERSION=latest
IMG_TAG=$(IMG_NAME):$(IMG_VERSION)
PORT=9900:9900

VOLDATA=`pwd`/data

all: $(IMG_NAME)

.PHONY: all


$(IMG_NAME):
	$(DOCKER) build -t $(IMG_TAG) --no-cache \
					-f Dockerfile .

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
	$(DOCKER) rm $(CTR_NAME)

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

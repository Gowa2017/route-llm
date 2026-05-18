IMAGE     ?= ccr.ccs.tencentyun.com/st-hub/llmrouter
TAG       ?= latest
PLATFORMS ?= linux/amd64,linux/arm64

.PHONY: build push

build:
	docker buildx build --load -t $(IMAGE):$(TAG) .

push:
	docker buildx build --platform $(PLATFORMS) --push -t $(IMAGE):$(TAG) .

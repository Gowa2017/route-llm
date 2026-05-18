IMAGE     ?= ccr.ccs.tencentyun.com/st-hub/llmrouter
TAG       ?= latest
PLATFORMS ?= linux/amd64,linux/arm64

.PHONY: build push build-push builder

builder:
	docker buildx inspect llmrouter-builder >/dev/null 2>&1 || \
		docker buildx create --name llmrouter-builder --driver docker-container --bootstrap

build: builder
	docker buildx build \
		--builder llmrouter-builder \
		--load \
		-t $(IMAGE):$(TAG) .

push: builder
	docker buildx build \
		--builder llmrouter-builder \
		--platform $(PLATFORMS) \
		--push \
		-t $(IMAGE):$(TAG) .

build-push: push

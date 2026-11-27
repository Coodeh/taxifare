include .env
export

GAR_IMAGE_NAME = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

.PHONY: run-api build-docker-local docker_build_local docker-run docker_build_cloud docker_push docker_deploy

run-api:
	uv run uvicorn api.fast:app --reload

build-docker-local:
	docker build -t $(DOCKER_IMAGE_NAME):dev .

docker_build_local: build-docker-local

docker-run:
	docker run --rm \
		-p 8001:8080 \
		--env-file .env \
		-v "$(HOME)/.config/gcloud/application_default_credentials.json:/tmp/keys/gackey.json:ro" \
		-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gackey.json \
		$(DOCKER_IMAGE_NAME):dev

docker_build_cloud:
	docker build --platform linux/amd64 \
		-t $(GAR_IMAGE_NAME):latest .

docker_push:
	docker push $(GAR_IMAGE_NAME):latest

docker_deploy:
	gcloud run deploy $(CLOUD_RUN_SERVICE_NAME) \
		--image $(GAR_IMAGE_NAME):latest \
		--allow-unauthenticated \
		--region $(GCP_REGION) \
		--project $(GCP_PROJECT) \
		--port 8080 \
		--memory 2Gi \
		--cpu 2 \
		--min-instances 0 \
		--max-instances 1 \
		--update-env-vars "MODEL_PATH=$(MODEL_PATH),RF_MODEL_PATH=$(RF_MODEL_PATH),KERAS_MODEL_PATH=$(KERAS_MODEL_PATH),PREPROCESSOR_PATH=$(PREPROCESSOR_PATH)"

-include .env
export

GAR_IMAGE_NAME = $(GCP_REGION)-docker.pkg.dev/$(GOOGLE_CLOUD_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

.PHONY: run-api build-docker-local docker-run docker_build_cloud docker_push docker_deploy

run-api:
	uv run uvicorn api.fast:app --reload


build-docker-local:
	docker build -t $(DOCKER_IMAGE_NAME):dev .

docker-run:
	docker run --rm -p 8001:8080 --env-file .env $(DOCKER_IMAGE_NAME):dev

docker_build_cloud:
	docker build --platform linux/amd64 -t $(GAR_IMAGE_NAME):latest .

docker_push:
	docker push $(GAR_IMAGE_NAME):latest

docker_deploy:
	gcloud run deploy $(CLOUD_RUN_SERVICE_NAME) \
		--image $(GAR_IMAGE_NAME):latest \
		--allow-unauthenticated \
		--region $(GCP_REGION) \
		--project $(GOOGLE_CLOUD_PROJECT) \
		--port 8080 \
		--memory 1Gi \
		--max-instances 1 \
		--update-env-vars MODEL_PATH=$(MODEL_PATH)

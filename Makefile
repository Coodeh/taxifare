-include .env
export

.PHONY: run-api build-docker-local docker-run

run-api:
	uv run uvicorn api.fast:app --reload


build-docker-local:
	docker build -t ${DOCKER_IMAGE_NAME}:dev .

docker-run:
	docker run --rm -p 8001:8080 --env-file .env ${DOCKER_IMAGE_NAME}:dev

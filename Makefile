.PHONY: devcontainer-build

devcontainer-build:
	docker compose -f .devcontainer/docker-compose.yml build text-encoders-devcontainer

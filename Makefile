.PHONY: devcontainer-build


devcontainer-build:
	[ -e .secrets/.env ] || touch .secrets/.env
	docker compose -f .devcontainer/docker-compose.yml build text-encoders-devcontainer

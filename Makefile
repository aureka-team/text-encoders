.PHONY: devcontainer-build


devcontainer-build:
	[ -e .secrets/.env ] || touch .secrets/.env
	docker compose -f .devcontainer/docker-compose.yml build text-encoders-devcontainer


weaviate-start:
	docker compose -f .devcontainer/docker-compose.yml up -d text-encoders-weaviate

weaviate-stop:
	docker compose -f .devcontainer/docker-compose.yml stop text-encoders-weaviate

weaviate-restart: -f .devcontainer/docker-compose.yml weaviate-stop weaviate-start

weaviate-flush: weaviate-stop
	$(info *** WARNING you are deleting all data from weaviate ***)
	sudo rm -r resources/db/weaviate
	docker compose up -d text-encoders-weaviate

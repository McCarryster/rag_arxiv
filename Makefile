DC := docker compose
IMAGE := weaviate_vector_db
DF := vector_db_service/Dockerfile

up: clean-containers clean-volumes build up-docker

clean-containers:
	docker rm -f $$(docker ps -a -q) || true

clean-volumes:
	docker volume rm -f $$(docker volume ls -q) || true

build:
	docker build -t $(IMAGE) -f $(DF) .

up-docker:
	$(DC) up -d

.PHONY: up clean-containers clean-volumes build up-docker

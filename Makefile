.PHONY: up

up:
	docker compose down -v --remove-orphans
	docker build -t index_service -f index_service/Dockerfile .
	docker compose up -d
.PHONY: up

up:
	docker compose down -v --remove-orphans
	docker build -t query_service -f query_service/Dockerfile .
	docker build -t index_service -f index_service/Dockerfile .
	docker build -t vector_db_service -f vector_db_service/Dockerfile .
	docker compose up -d
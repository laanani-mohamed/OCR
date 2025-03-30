.PHONY: build run stop restart logs clean

build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

clean:
	docker compose down -v


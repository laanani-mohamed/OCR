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
	docker-compose down -v


# tester api du pdf
curl -X 'POST' \
  'http://localhost:9876/process_pdf/' \
  -F 'file=@/Users/mac/Desktop/Qalqul/OCR/files_test/consta_1 (dragged).pdf'


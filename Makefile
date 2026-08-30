.PHONY: install lint format test api ui replay docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

api:
	uv run uvicorn abtestlab.api.main:app --reload --port 8240

ui:
	ABTESTLAB_API_URL=http://localhost:8240 uv run streamlit run src/abtestlab/ui/app.py --server.port 8741

replay:
	uv run abtestlab monitor data/example_looks.csv

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

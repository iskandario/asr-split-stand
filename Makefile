# Стенд «распознавание речи: браузер или сервер». Все команды — из корня репозитория.
# Нужны: Docker с Compose v2, uv, Node.js 20+ (только для юнит-тестов планировщика).
.PHONY: help data build up smoke test aa experiment repeat analyze reproduce ci down clean-results

UV ?= uv
RUN := $(UV) run --frozen
# на macOS не даём ноутбуку уснуть во время длинной серии
KEEPAWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -dimsu")
RAW ?= results/raw
OUT ?= results/processed

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  make %-14s %s\n", $$1, $$2}'

data: ## скачать фрагменты Golos и сверить SHA-256 с data/manifest.csv
	$(RUN) python data/prepare.py
	$(RUN) python data/prepare.py --check

build: ## собрать образы сервера и клиента (модели скачиваются по ревизиям)
	docker compose --profile run build

up: ## поднять сервер распознавания и Prometheus
	docker compose up -d --wait server prometheus

test: ## юнит-тесты планировщика (Node) и анализа/раннера (pytest)
	node --test tests/
	$(RUN) pytest -q

smoke: up ## проверка стенда целиком: сервер отвечает, метрики есть, 2 коротких прогона проходят
	curl -sf localhost:8000/health
	curl -sf -X POST --data-binary @data/audio/tune_000.wav -H 'content-type: audio/wav' localhost:8000/transcribe
	curl -sf localhost:8000/metrics | grep -q asr_inference_seconds_count
	curl -sf 'localhost:9090/api/v1/query?query=up' | grep -q '"value"'
	$(RUN) python runner/run_experiment.py configs/smoke.yaml
	@echo "smoke: OK"

aa: up ## A/A-тест: 10 прогонов одной конфигурации (~25 мин)
	$(KEEPAWAKE) $(RUN) python runner/run_experiment.py configs/aa.yaml

experiment: up ## основной эксперимент: 252 прогона (~12 ч), продолжает с места остановки
	$(KEEPAWAKE) $(RUN) python runner/run_experiment.py configs/experiment.yaml

analyze: ## таблицы, статистика и графики: $(RAW) -> $(OUT)
	$(RUN) python analysis/analyze.py --raw $(RAW) --out $(OUT)

reproduce: data build test smoke aa experiment analyze ## всё с нуля одной командой

ci: up ## сокращённый прогон для CI
	$(RUN) python runner/run_experiment.py configs/ci.yaml
	$(RUN) python analysis/analyze.py --experiment ci --out results/ci || true

down: ## остановить стенд
	docker compose --profile run down

clean-results: ## удалить сырые результаты (осторожно)
	rm -rf results/raw/* results/processed/*

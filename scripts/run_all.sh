#!/bin/sh
# Полная серия: ждём питание от сети -> A/A -> основной эксперимент -> анализ. Лог: results/run_all.log
cd "$(dirname "$0")/.."
while ! pmset -g batt | grep -q "AC Power"; do echo "$(date '+%F %T') жду подключения зарядки"; sleep 30; done
echo "$(date '+%F %T') питание от сети, старт"
caffeinate -dimsu uv run --frozen python runner/run_experiment.py configs/aa.yaml
caffeinate -dimsu uv run --frozen python runner/run_experiment.py configs/experiment.yaml
# повтор после сбоев (продолжает с места остановки, failed-прогоны перезапускаются)
caffeinate -dimsu uv run --frozen python runner/run_experiment.py configs/experiment.yaml
uv run --frozen python analysis/analyze.py
echo "$(date '+%F %T') ВСЁ ГОТОВО"

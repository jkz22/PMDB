# Full pipeline. Use the project venv's python.
PY := .venv/Scripts/python.exe
ifeq ($(wildcard $(PY)),)
PY := python
endif

.PHONY: all qc kpis stats plots

all: stats plots

qc:
	$(PY) scripts/qc_load.py
	$(PY) scripts/qc_segment.py
	$(PY) scripts/qc_instances.py

kpis: qc
	$(PY) src/run.py

stats: kpis
	$(PY) src/stats.py

plots: kpis
	$(PY) scripts/plot_kpis.py

# Full pipeline. Use the project venv's python.
PY := .venv/Scripts/python.exe
ifeq ($(wildcard $(PY)),)
PY := python
endif

.PHONY: all qc kpis stats

all: qc kpis stats

qc:
	$(PY) scripts/qc_load.py
	$(PY) scripts/qc_segment.py
	$(PY) scripts/qc_instances.py

kpis:
	$(PY) src/run.py

stats:
	$(PY) src/stats.py

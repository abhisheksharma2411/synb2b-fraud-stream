PY ?= python3
export PYTHONPATH := src

.PHONY: help venv test reproduce-small reproduce-full paper-data drift-params \
        summary audit paper refcheck numbers clean

help:
	@echo "make venv             create .venv and install the pinned requirements"
	@echo "make test             pytest"
	@echo "make reproduce-small  reduced-size end-to-end pass (target: under 4 minutes)"
	@echo "make reproduce-full   the full grid: 6 methods + oracle, 2 streams, 5 seeds,"
	@echo "                      9 ablation factors, 10k bootstrap resamples"
	@echo "make paper-data       regenerate results/paper_data.tex and splice it into"
	@echo "                      paper/main.tex between the INLINE-DATA markers"
	@echo "make drift-params     re-solve the drift constants (writes drift_params.json)"
	@echo "make refcheck         re-verify every citation against arXiv/Crossref/Zenodo"
	@echo "make numbers          re-derive every printed number from results.json"
	@echo "make audit            run the prose gates over paper/main.tex"
	@echo "make paper            pdflatex both anonymisation settings, report page counts"

venv:
	$(PY) -m venv .venv && ./.venv/bin/python -m pip install -U pip \
		&& ./.venv/bin/python -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests -q

reproduce-small:
	$(PY) -m t13.experiment --small
	$(PY) tools/make_summary.py --small

reproduce-full:
	$(PY) -m t13.experiment
	$(PY) tools/make_summary.py

paper-data:
	$(PY) tools/make_paper_data.py
	$(PY) tools/inline_paper_data.py

drift-params:
	$(PY) tools/calibrate_drift.py

refcheck:
	$(PY) tools/refcheck_arxiv.py

numbers:
	$(PY) tools/check_numbers.py

audit:
	$(PY) tools/audit_prose.py paper/main.tex

paper:
	$(PY) tools/build_paper.py

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.pdf
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: all prepare features train evaluate report test lint clean

PYTHON ?= /home/tales/miniforge3/envs/improve-traffic/bin/python
QUARTO ?= /usr/lib/rstudio/resources/app/bin/quarto/bin/quarto
export PYTHONPATH := .

all: prepare features train evaluate report

prepare:
	$(PYTHON) src/data_prep.py --train data/train.csv --test data/test.csv --weather data/weather.csv --out-dir data/processed/

features:
	$(PYTHON) src/features.py --in-dir data/processed/ --out-dir data/processed/

train:
	$(PYTHON) src/train.py --in-dir data/processed/ --models-dir models/ --eval-date 2016-06-16

evaluate:
	$(PYTHON) src/evaluate.py --models-dir models/ --features-dir data/processed/ --reports-dir reports/

report:
	XDG_RUNTIME_DIR=/tmp XDG_DATA_HOME=/tmp XDG_CACHE_HOME=/tmp XDG_CONFIG_HOME=/tmp QUARTO_PYTHON=$(PYTHON) $(QUARTO) render report/

test:
	$(PYTHON) -m unittest discover tests/

lint:
	ruff check src/ tests/

clean:
	rm -rf data/processed/*.parquet data/processed/submission.csv
	rm -rf models/*.joblib models/*.json
	rm -rf reports/figures/*.png reports/figures/*.svg
	rm -rf report/_site/ .quarto/

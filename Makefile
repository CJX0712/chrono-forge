.PHONY: install test demo benchmark lock clean

install:
	pip install -r requirements.txt

test:
	pytest -q -W ignore::UserWarning

demo:
	python examples/run_demo.py

benchmark:
	python -m chrono_forge.cli benchmark --out benchmark.json

lock:
	pip freeze > requirements.lock.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

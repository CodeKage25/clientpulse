all:
	python -m clientpulse.pipeline
	python -m automation.run
	pytest -q

serve:
	uvicorn clientpulse.api:app --reload

test:
	pytest -q

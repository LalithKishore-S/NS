test:
	PYTHONPATH=. python -m pytest -q tests

dummy:
	PYTHONPATH=. python -c "from common.contracts import ExperimentConfig; from common.dummy import make_dummy_artifacts; make_dummy_artifacts(ExperimentConfig(run_id='dummy'))"

experiments:
	PYTHONPATH=. python run_experiments.py

.PHONY: test dummy experiments

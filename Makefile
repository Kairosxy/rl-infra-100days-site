.PHONY: test lint day-01 day-02 day-03 day-04 reference

test:
	@set -e; for test_dir in examples/day_*; do \
		if [ -d "$$test_dir" ]; then \
			echo "==> $$test_dir"; \
			python -m unittest discover -s "$$test_dir" -p 'test_*.py' -v; \
		fi; \
	done
	python -m unittest discover -s reference_system -p 'test_*.py' -v

lint:
	python -m ruff check examples reference_system

day-01:
	python examples/day_01/trajectory_contract.py

day-02:
	python examples/day_02/memory_budget.py

day-03:
	python examples/day_03/gradient_accumulation_lab.py

day-04:
	python examples/day_04/policy_gradient_lab.py
	python examples/day_04/sequence_logprob_lab.py

reference:
	python reference_system/toy_reinforce.py

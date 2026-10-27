.PHONY: test lint lint-prose day-01 day-02 day-03 day-04 reference

test:
	@set -e; for test_dir in examples/day_*; do \
		if [ -d "$$test_dir" ]; then \
			echo "==> $$test_dir"; \
			python -m unittest discover -s "$$test_dir" -p 'test_*.py' -v; \
		fi; \
	done
	python -m unittest discover -s reference_system -p 'test_*.py' -v

lint:
	python -m ruff check examples reference_system scripts

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

# 读者页面里的元话语：在说文档自己、或否定读者没见过的方案。规则见 STYLE.md。
PROSE_PATTERNS := 不是大模型训练的测量结果|不是实测|你现在能验证的结论|真正标志|而不是拼成|这次不把|完全自愿|自由选课|不代表学习者|不用于解锁|不再强制|不需要登录
PROSE_PATHS := README.md knowledge/topics knowledge/cases knowledge/implementations quiz/README.md

lint-prose:
	@if grep -rnE "$(PROSE_PATTERNS)" $(PROSE_PATHS); then \
		echo "发现元话语，按 STYLE.md 改写"; exit 1; \
	fi

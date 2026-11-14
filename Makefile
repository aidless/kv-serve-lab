# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: ci.yml
# 命令逐字取自该来源，未增删任何参数。
# 归一: python → python3（本机无 python 别名，实测 Error 127）
.PHONY: eval

eval:
	python3 -m pytest -q

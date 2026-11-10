#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "이 스크립트는 Apple Silicon Mac용입니다." >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  python_bin="${PYTHON_BIN:-python3.12}"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python 3.12가 필요합니다. 설치 후 PYTHON_BIN으로 실행 파일을 지정해주세요." >&2
    exit 1
  fi
  "$python_bin" -m venv .venv
fi
.venv/bin/python - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12) or platform.machine() != "arm64":
    raise SystemExit(".venv는 ARM64 Python 3.12로 만들어야 합니다. 기존 환경을 먼저 확인해주세요.")
PY
export KERAS_BACKEND=tensorflow
export CUDA_VISIBLE_DEVICES=-1
export KERAS_HOME="$PWD/.venv/.keras"
export MPLCONFIGDIR="$PWD/.venv/.matplotlib"
.venv/bin/python -m pip install --index-url https://pypi.org/simple --cache-dir "$PWD/.venv/.pip-cache" -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/check_environment.py --output .venv/environment.json
.venv/bin/python -m pip freeze > .venv/requirements-macos.txt
echo "환경 점검 완료. source .venv/bin/activate 로 활성화하세요."

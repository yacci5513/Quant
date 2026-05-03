#!/usr/bin/env bash
# pre-commit hook: 계좌/거래 데이터 파일 커밋 차단
set -euo pipefail

violations=$(git diff --cached --name-only \
    | grep -iE '(account|portfolio|balance|trades?|orders?|positions?)_.*\.(csv|json|parquet)$' || true)

if [ -n "$violations" ]; then
    echo "❌ ERROR: 계좌/거래 데이터 파일 커밋 금지"
    echo "차단된 파일:"
    echo "$violations" | sed 's/^/  - /'
    echo ""
    echo "조치: data/ 하위는 .gitignore에 등록되어 있음. 그 외 위치라면 이동 후 재시도"
    exit 1
fi

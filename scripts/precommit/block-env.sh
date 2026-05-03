#!/usr/bin/env bash
# pre-commit hook: .env* 파일 커밋 차단 (.env.example만 허용)
set -euo pipefail

violations=$(git diff --cached --name-only \
    | grep -E '(^|/)\.env$|(^|/)\.env\.[^.]+$' \
    | grep -v '\.env\.example$' || true)

if [ -n "$violations" ]; then
    echo "❌ ERROR: .env* 파일은 커밋 금지 (.env.example 만 허용)"
    echo "차단된 파일:"
    echo "$violations" | sed 's/^/  - /'
    echo ""
    echo "조치: git rm --cached <파일>  으로 staged에서 제거"
    exit 1
fi

#!/usr/bin/env bash
# pre-commit hook: 토큰/세션/시크릿 파일 커밋 차단
set -euo pipefail

violations=$(git diff --cached --name-only \
    | grep -iE '(token|session|credential|secret|private[_-]?key)\.(json|yaml|yml|txt|env)$' || true)

if [ -n "$violations" ]; then
    echo "❌ ERROR: 토큰/시크릿 파일 커밋 금지"
    echo "차단된 파일:"
    echo "$violations" | sed 's/^/  - /'
    echo ""
    echo "조치: 환경변수 또는 .env로 옮기고 git rm --cached 로 staged에서 제거"
    exit 1
fi

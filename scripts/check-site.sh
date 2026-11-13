#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== Building original Satellite-themed Jekyll site =="
bundle exec jekyll build
for f in _site/index.html _site/assets/css/style.css _site/assets/js/common.js _site/assets/img/profile.webp _site/Statistics/correlation-and-p-values.html; do
  if [[ ! -f "$f" ]]; then echo "MISSING: $f" >&2; exit 1; fi
  echo "OK: $f"
done
if grep -Eq 'Category A|Category B|/notebooks/.*ipynb' _site/index.html; then
  echo 'Inspect homepage: unexpected demo names or notebook link' >&2; exit 1
fi
echo 'Static build checks passed. Preview at localhost:4000 before pushing.'

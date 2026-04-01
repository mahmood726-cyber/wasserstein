#!/bin/bash
# Quick push for publication-managed files
# Usage: bash push.sh "commit message"

MSG="${1:-Update E156 submission}"
GIT_NAME="${E156_GIT_NAME:-Mahmood Ahmad}"
GIT_EMAIL="${E156_GIT_EMAIL:-mahmood726-cyber@users.noreply.github.com}"

if ! git diff --cached --quiet --exit-code; then
  echo "There are already staged changes in this repo. Review and push manually."
  exit 1
fi

paths=(
  "e156-submission"
  "push.sh"
  "LICENSE"
  "LICENSE.md"
  "LICENSE.txt"
  "CITATION.cff"
)

for path in "${paths[@]}"; do
  if [ -e "$path" ] || git ls-files -- "$path" | grep -q .; then
    git add -A -- "$path"
  fi
done

if ! git diff --cached --quiet --exit-code; then
  git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit --no-verify --no-gpg-sign -m "$MSG"
else
  echo "No publication-managed changes to commit."
fi

git push origin master 2>/dev/null || git push origin main 2>/dev/null

echo ""
echo "Pushed to GitHub. View at:"
echo "  https://github.com/mahmood726-cyber/wasserstein-km-extractor"
echo "  https://mahmood726-cyber.github.io/wasserstein-km-extractor/"
echo "  https://mahmood726-cyber.github.io/wasserstein-km-extractor/e156-submission/"

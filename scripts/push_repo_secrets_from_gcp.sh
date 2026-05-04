#!/usr/bin/env bash
# Push 3 secrets from GCP Secret Manager (thet-integration-af) into
# AssessorFlow-ISS/classifier-agent as GitHub Actions repo secrets.
#
# Required for the e2e-golden-pipeline.yml workflow.
#
# Prereqs:
#   - gcloud authenticated to the user account that can read secrets in
#     the thet-integration-af project (run `gcloud auth login` first)
#   - gh authenticated to a user with write access to the repo's secrets
#     (run `gh auth status` to verify; needs `repo` + `admin:repo_hook` scopes)
#
# Usage:
#   bash scripts/push_repo_secrets_from_gcp.sh [REPO]
#   REPO defaults to AssessorFlow-ISS/classifier-agent
#
# To inspect actual GCP secret names first (in case the defaults below
# don't match your project's naming convention):
#
#   gcloud secrets list --project=thet-integration-af --format='value(name)'
#
# Then edit the GCP_* variables below to match.

set -euo pipefail

REPO="${1:-AssessorFlow-ISS/classifier-agent}"
GCP_PROJECT="thet-integration-af"

# ── GCP secret name  →  GitHub repo secret name ─────────────────────────
# Edit these mappings to match your GCP secret naming.
declare -A SECRET_MAP=(
  ["orchestrator-db-password"]="ORCHESTRATOR_DB_PASSWORD"
  ["langfuse-public-key"]="LANGFUSE_PUBLIC_KEY"
  ["langfuse-secret-key"]="LANGFUSE_SECRET_KEY"
)

# ── Prereq checks ───────────────────────────────────────────────────────
echo "▶ Checking prereqs..."
command -v gcloud >/dev/null || { echo "✗ gcloud CLI not found"; exit 1; }
command -v gh >/dev/null     || { echo "✗ gh CLI not found"; exit 1; }

if ! gcloud auth print-access-token >/dev/null 2>&1; then
  echo "✗ gcloud not authenticated. Run: gcloud auth login"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "✗ gh not authenticated. Run: gh auth login"
  exit 1
fi

echo "✓ gcloud + gh authenticated"
echo "  Target repo: ${REPO}"
echo "  Source GCP project: ${GCP_PROJECT}"
echo ""

# ── Verify each GCP secret exists ───────────────────────────────────────
echo "▶ Verifying source secrets in GCP..."
missing=()
for gcp_name in "${!SECRET_MAP[@]}"; do
  if gcloud secrets describe "$gcp_name" --project="$GCP_PROJECT" >/dev/null 2>&1; then
    echo "  ✓ ${gcp_name} → ${SECRET_MAP[$gcp_name]}"
  else
    echo "  ✗ ${gcp_name} NOT FOUND in ${GCP_PROJECT}"
    missing+=("$gcp_name")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  echo "✗ Missing GCP secrets: ${missing[*]}"
  echo ""
  echo "Available secrets in ${GCP_PROJECT}:"
  gcloud secrets list --project="$GCP_PROJECT" --format='value(name)' | sed 's/^/  /'
  echo ""
  echo "Edit the SECRET_MAP at the top of this script to match the actual names, then re-run."
  exit 1
fi
echo ""

# ── Pull each secret value and push to GH ────────────────────────────────
echo "▶ Pushing secrets to ${REPO}..."
for gcp_name in "${!SECRET_MAP[@]}"; do
  gh_name="${SECRET_MAP[$gcp_name]}"
  printf "  %s → %s ... " "$gcp_name" "$gh_name"

  # Pull from GCP Secret Manager (latest version)
  value=$(gcloud secrets versions access latest \
    --secret="$gcp_name" \
    --project="$GCP_PROJECT" 2>/dev/null) || {
      echo "FAIL (could not access secret value)"
      exit 1
    }

  # Push to GH repo secret (overwrites if exists)
  printf '%s' "$value" | gh secret set "$gh_name" -R "$REPO" --body - >/dev/null
  echo "OK"
done
echo ""

# ── Verify what GH now reports ──────────────────────────────────────────
echo "▶ Verifying repo secrets on GitHub..."
gh secret list -R "$REPO" | grep -E "ORCHESTRATOR_DB_PASSWORD|LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY" | sed 's/^/  /' || true
echo ""
echo "✓ Done. The e2e-golden-pipeline.yml workflow can now run."
echo ""
echo "Next: trigger a smoke run with:"
echo "  gh workflow run \"E2E — Golden Pipeline (real validator-write + KS + classifier + …)\" -R ${REPO} --ref main -f scenario=insufficient"

#!/usr/bin/env bash
set -euo pipefail

# Stages only public positive-control inputs.  It deliberately neither downloads
# the model nor creates/reads the private answer key.  The destination must be
# new so an interrupted or altered staging cannot silently become evidence.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${SENTRY_PUBLIC_SOURCE_ROOT:?set a fresh destination}"
[[ ! -e "$DESTINATION" ]] || { echo "refusing to overwrite: $DESTINATION" >&2; exit 2; }
mkdir -p "$DESTINATION"

UPSTREAM_COMMIT="db04f4150edf940559b5f3147f65d808e9313efd"
OFFLINE_COMMIT="68caa7ae93840af798d259fdbcfbbd0667aa9a09"
NUMBERS_REVISION="4fa5997d83c88edf028e2c13c7107baa731eb30c"
CODE_REVISION="95020687463ac33c430202a923fe47756b82d377"

git clone --no-checkout https://github.com/MinhxLe/subliminal-learning.git "$DESTINATION/upstream"
git -C "$DESTINATION/upstream" checkout --detach "$UPSTREAM_COMMIT"
git clone --no-checkout https://github.com/iremkrc/subliminal-learning-open.git "$DESTINATION/offline_replication"
git -C "$DESTINATION/offline_replication" checkout --detach "$OFFLINE_COMMIT"

command -v hf >/dev/null || { echo "missing Hugging Face CLI 'hf'" >&2; exit 2; }
hf download --repo-type dataset --revision "$NUMBERS_REVISION" \
  minhxle/subliminal-learning_numbers_dataset --local-dir "$DESTINATION/numbers"
hf download --repo-type dataset --revision "$CODE_REVISION" \
  minhxle/subliminal-learning_code_dataset --local-dir "$DESTINATION/code"

(cd "$DESTINATION" && find upstream offline_replication numbers code -type f -print0 | sort -z | xargs -0 sha256sum) \
  > "$DESTINATION/public_sources.sha256"
cp "$ROOT/configs/sentry_g0.yaml" "$DESTINATION/public_contract.yaml"
sha256sum "$DESTINATION/public_contract.yaml" >> "$DESTINATION/public_sources.sha256"
echo "SENTRY public sources staged at $DESTINATION"

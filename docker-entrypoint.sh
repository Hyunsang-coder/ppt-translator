#!/bin/sh
# Fetch secrets from AWS SSM Parameter Store and export them as environment
# variables, then exec the main process. The EC2 instance's IAM role grants
# read+decrypt access to /ppt-translator/* SecureString params, so no API
# keys ever live on disk.
#
# If SSM_PARAM_PREFIX is unset, secret fetching is skipped entirely and any
# already-present env vars (e.g. local .env via docker run -e) are used as-is.
# This keeps local development working without AWS.
set -e

if [ -n "${SSM_PARAM_PREFIX}" ]; then
    echo "[entrypoint] Loading secrets from SSM prefix ${SSM_PARAM_PREFIX}"
    # Capture the export lines first so a fetch failure aborts startup —
    # `eval "$(...)"` would otherwise swallow the script's non-zero exit and
    # let the container boot with no/old keys.
    if ! ssm_exports="$(python /app/fetch_ssm.py)"; then
        echo "[entrypoint] FATAL: failed to load secrets from SSM; refusing to start" >&2
        exit 1
    fi
    eval "${ssm_exports}"
    echo "[entrypoint] Secrets loaded from SSM"
else
    echo "[entrypoint] SSM_PARAM_PREFIX unset; using existing environment"
fi

exec "$@"

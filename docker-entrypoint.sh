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
    # fetch_ssm.py prints `KEY=value` lines for each parameter under the prefix.
    eval "$(python /app/fetch_ssm.py)"
    echo "[entrypoint] Secrets loaded from SSM"
else
    echo "[entrypoint] SSM_PARAM_PREFIX unset; using existing environment"
fi

exec "$@"

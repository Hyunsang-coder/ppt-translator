"""Fetch SecureString parameters from AWS SSM and print shell `export` lines.

Reads every parameter under ``$SSM_PARAM_PREFIX`` (default ``/ppt-translator``)
and maps the last path segment (uppercased) to an environment variable, e.g.
``/ppt-translator/openai_api_key`` -> ``OPENAI_API_KEY``.

Output is consumed via ``eval "$(python fetch_ssm.py)"`` in the entrypoint.
Values are shell-quoted so arbitrary characters cannot break out of the
assignment. Failures are fatal (exit non-zero) so the container does not start
silently without its keys.
"""

from __future__ import annotations

import os
import shlex
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

PREFIX = os.environ.get("SSM_PARAM_PREFIX", "/ppt-translator")
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-northeast-2"


def main() -> int:
    client = boto3.client("ssm", region_name=REGION)
    params = {}
    try:
        paginator = client.get_paginator("get_parameters_by_path")
        for page in paginator.paginate(Path=PREFIX, Recursive=True, WithDecryption=True):
            for p in page["Parameters"]:
                # /ppt-translator/openai_api_key -> OPENAI_API_KEY
                name = p["Name"].rstrip("/").rsplit("/", 1)[-1].upper()
                params[name] = p["Value"]
    except (BotoCoreError, ClientError) as exc:
        print(f"[fetch_ssm] Failed to read SSM parameters under {PREFIX}: {exc}", file=sys.stderr)
        return 1

    if not params:
        print(f"[fetch_ssm] No parameters found under {PREFIX}", file=sys.stderr)
        return 1

    for key, value in params.items():
        sys.stdout.write(f"export {key}={shlex.quote(value)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

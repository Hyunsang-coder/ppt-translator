"""Entrypoint for the bundled local sidecar.

The Tauri shell spawns this process with a free port it picked, injects the
user's API keys as environment variables (read from the OS keychain), and waits
for the ``SIDECAR_READY`` line on stdout before pointing the WebView at the
server.

This is intentionally separate from ``api.py``'s ``__main__`` block, which stays
as the legacy server entrypoint (hardcoded host/port for EC2).
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

# api.py lives at the repo root; PyInstaller bundles it alongside this file.
from api import app


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPT translator local sidecar")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, loopback only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind. 0 lets the OS choose a free port.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # Bind the socket up front so we can report the actual port even when the
    # caller passed --port 0 (OS-assigned). uvicorn can run on a preopened sock.
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    actual_port = sock.getsockname()[1]

    # Hand the bound port to the Rust shell. It blocks on this line.
    print(f"SIDECAR_READY port={actual_port}", flush=True)

    config = uvicorn.Config(app, log_level="info")
    server = uvicorn.Server(config)
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()

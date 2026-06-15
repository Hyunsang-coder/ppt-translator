"""Entrypoint for the bundled local sidecar.

The Tauri shell spawns this process with a free port it picked, injects the
user's API keys as environment variables (read from the OS keychain), and waits
for the ``SIDECAR_READY`` line on stdout before pointing the WebView at the
server.

This is intentionally separate from ``api.py``'s ``__main__`` block, which stays
as a direct local development entrypoint.
"""

from __future__ import annotations

import argparse
import sys

# NOTE: `api` and `uvicorn` are intentionally NOT imported at module top level.
# Importing the FastAPI app pulls in langchain and a large dependency tree,
# which can take ~20s as a PyInstaller onedir bundle cold-starts. We bind the
# socket and print SIDECAR_READY *first*, so the Rust shell learns the port
# immediately and can poll /health while the heavy import finishes.


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

    # Hand the bound port to the Rust shell BEFORE the heavy import, so it can
    # start polling /health right away instead of blocking ~20s on import.
    print(f"SIDECAR_READY port={actual_port}", flush=True)

    # Now do the expensive imports.
    import uvicorn

    from api import app

    config = uvicorn.Config(app, log_level="info")
    server = uvicorn.Server(config)
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()

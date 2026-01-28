#!/usr/bin/env python3
"""Entry point for the rt CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Iterable, List

from orchestrator import Orchestrator

try:
    from _version import __version__
except Exception:  # pragma: no cover - fallback for missing file
    __version__ = "0.0.0"


INSTALL_URL = "https://raw.githubusercontent.com/ryangerardwilson/rt/main/install.sh"
LATEST_RELEASE_API = "https://api.github.com/repos/ryangerardwilson/rt/releases/latest"


def _version_tuple(version: str) -> tuple[int, ...]:
    if not version:
        return (0,)
    version = version.strip()
    if version.startswith("v"):
        version = version[1:]
    parts: List[int] = []
    for segment in version.split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _is_version_newer(candidate: str, current: str) -> bool:
    cand_tuple = _version_tuple(candidate)
    curr_tuple = _version_tuple(current)
    length = max(len(cand_tuple), len(curr_tuple))
    cand_tuple += (0,) * (length - len(cand_tuple))
    curr_tuple += (0,) * (length - len(curr_tuple))
    return cand_tuple > curr_tuple


def _get_latest_version(timeout: float = 5.0) -> str | None:
    try:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError
    except ImportError:  # pragma: no cover - extremely unlikely
        return None

    try:
        request = Request(LATEST_RELEASE_API, headers={"User-Agent": "rt-updater"})
        with urlopen(request, timeout=timeout) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError):
        return None

    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None

    tag = payload.get("tag_name") or payload.get("name")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    return None


def _run_upgrade() -> int:
    try:
        curl = subprocess.Popen(
            ["curl", "-fsSL", INSTALL_URL],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("Upgrade requires curl", file=sys.stderr)
        return 1

    try:
        bash = subprocess.Popen(["bash"], stdin=curl.stdout)
        if curl.stdout is not None:
            curl.stdout.close()
    except FileNotFoundError:
        print("Upgrade requires bash", file=sys.stderr)
        curl.terminate()
        curl.wait()
        return 1

    bash_rc = bash.wait()
    curl_rc = curl.wait()

    if curl_rc != 0:
        stderr = (
            curl.stderr.read().decode("utf-8", errors="replace") if curl.stderr else ""
        )
        if stderr:
            sys.stderr.write(stderr)
        return curl_rc

    return bash_rc


def _print_help() -> None:
    print("Usage: rt [options]\n")
    print("Global options:")
    print("  -v, --version       Print the installed version and exit")
    print("  -u, --update        Upgrade rt to the latest release")
    print("  -h, --help          Show this help and doc-mode options")
    print()
    print("Doc-mode and course management options:")
    orchestrator = Orchestrator(argv=["--help"])
    orchestrator._print_help()


def _handle_global_flags(argv: Iterable[str]) -> List[str]:
    remaining: List[str] = []
    for token in argv:
        if token in {"-v", "-V", "--version"}:
            print(__version__)
            raise SystemExit(0)
        if token in {"-u", "--update"}:
            latest = _get_latest_version()
            if latest is None:
                print(
                    "Unable to determine latest version; attempting upgrade…",
                    file=sys.stderr,
                )
                raise SystemExit(_run_upgrade())

            if __version__ and __version__ != "0.0.0" and not _is_version_newer(
                latest, __version__
            ):
                print(f"Already running the latest version ({__version__}).")
                raise SystemExit(0)

            if __version__ and __version__ != "0.0.0":
                print(f"Upgrading from {__version__} to {latest}…")
            else:
                print(f"Upgrading to {latest}…")
            raise SystemExit(_run_upgrade())
        if token in {"-h", "--help"}:
            _print_help()
            raise SystemExit(0)
        remaining.append(token)
    return remaining


def main() -> None:
    os.environ.setdefault("ESCDELAY", "25")
    os.environ.setdefault("TERM", "xterm-256color")

    argv = list(sys.argv[1:])
    filtered = _handle_global_flags(argv)
    Orchestrator(argv=filtered).run()


if __name__ == "__main__":
    main()

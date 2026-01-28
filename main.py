#!/usr/bin/env python3
"""Entry point for the rt CLI."""

from orchestrator import Orchestrator


def main() -> None:
    Orchestrator().run()


if __name__ == "__main__":
    main()

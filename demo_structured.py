#!/usr/bin/env python3
"""Example script that requests a structured joke from Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

# Ensure repository root is importable when running from this subdirectory.
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ollama import OllamaClient, OllamaStructuredError


@dataclass
class Joke:
    setup: str
    punchline: str


def main() -> None:
    client = OllamaClient(model="gpt-oss:120b")
    prompt = "Tell a short, clean joke as JSON with 'setup' and 'punchline'."
    try:
        joke = client.run(prompt, Joke)
    except OllamaStructuredError as exc:
        print(f"Failed to fetch joke: {exc}")
        return

    print("Structured joke from Ollama:")
    print(f"  Setup    : {joke.setup}")
    print(f"  Punchline: {joke.punchline}")


if __name__ == "__main__":
    main()

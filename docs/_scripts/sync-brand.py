#!/usr/bin/env python3
"""Sync _brand.yml from ggsegverse.github.io before Quarto render."""

import urllib.request
from pathlib import Path

BRAND_URL = "https://raw.githubusercontent.com/ggsegverse/ggsegverse.github.io/main/_brand.yml"
DOCS_DIR = Path(__file__).parent.parent
OUTPUT_FILE = DOCS_DIR / "_brand.yml"

HEADER = """# ggsegpy brand - synced from ggsegverse.github.io
# Source: {url}
# Auto-synced by _scripts/sync-brand.py

""".format(url=BRAND_URL)

EXCLUDED_SECTIONS = ["logo"]


def fetch_brand():
    with urllib.request.urlopen(BRAND_URL) as response:
        return response.read().decode("utf-8")


def filter_brand(content: str) -> str:
    """Remove sections that reference local files (logos, custom fonts)."""
    import re

    lines = content.split("\n")
    result = []
    skip_section = False
    current_indent = 0

    for line in lines:
        if not line.strip():
            if not skip_section:
                result.append(line)
            continue

        indent = len(line) - len(line.lstrip())

        for section in EXCLUDED_SECTIONS:
            if line.strip().startswith(f"{section}:"):
                skip_section = True
                current_indent = indent
                break
        else:
            if skip_section:
                if indent <= current_indent and line.strip():
                    skip_section = False
                    result.append(line)
            else:
                result.append(line)

    font_lines = []
    in_fonts = False
    for i, line in enumerate(result):
        if "family: Nelphim" in line:
            j = i
            while j >= 0 and "- family:" not in result[j]:
                j -= 1
            start = j
            j = i + 1
            while j < len(result) and (result[j].strip().startswith("-") is False or "family:" not in result[j]):
                if result[j].strip() and not result[j].strip().startswith("-") and ":" in result[j]:
                    j += 1
                else:
                    break
            font_lines.extend(range(start, j))

    result = [line for i, line in enumerate(result) if i not in font_lines]

    return "\n".join(result)


def main():
    print(f"Fetching brand.yml from {BRAND_URL}")
    content = fetch_brand()

    filtered = filter_brand(content)

    with open(OUTPUT_FILE, "w") as f:
        f.write(HEADER)
        f.write(filtered)

    print(f"Updated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

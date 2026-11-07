"""Chapter 00: repository setup smoke tests."""

from pathlib import Path

import kalman

ROOT = Path(__file__).resolve().parents[1]


def test_package_version():
    assert kalman.__version__ == "0.1.0"


def test_book_sources_are_utf8():
    for path in (ROOT / "book").glob("*.md"):
        path.read_text(encoding="utf-8")  # raises UnicodeDecodeError if not UTF-8


def test_no_ipynb_in_book():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*.ipynb" in [line.strip() for line in ignored]

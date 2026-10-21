"""Thin wrapper around ``aldwknn.data.main`` (download + validate UCI datasets).

ucimlrepo 0.0.7 exposes ``fetch_ucirepo`` instead of the top-level
``fetch_uci_dataset`` name that ``aldwknn.data`` imports, so a local alias is
installed here before delegating to the package (the package itself is not
modified).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ucimlrepo  # noqa: E402
from ucimlrepo.fetch import fetch_ucirepo  # noqa: E402

if not hasattr(ucimlrepo, "fetch_uci_dataset"):
    ucimlrepo.fetch_uci_dataset = fetch_ucirepo

from aldwknn.data import main  # noqa: E402


def cli() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(cli())

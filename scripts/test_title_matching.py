#!/usr/bin/env python3
"""Quick smoke test for title matching."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from bookrec.title_matching import TitleMatcher, normalize_title_core, normalize_title_for_match


def main() -> None:
    t1 = "Harry Potter and the Half-Blood Prince (Harry Potter, #6)"
    print("norm:", normalize_title_for_match(t1))
    print("core:", normalize_title_core(t1))

    cat = pd.DataFrame(
        {
            "id": [1, 2],
            "name": [
                "Harry Potter and the Half-Blood Prince (Harry Potter, #6)",
                "The Restaurant at the End of the Universe (Hitchhiker's Guide to the Galaxy, #2)",
            ],
        }
    )
    matcher = TitleMatcher.from_catalog(cat, fuzzy_threshold=88)
    titles = pd.Series(
        [
            "harry potter and the half-blood prince",
            "The Restaurant at the End of the Universe",
            "Totally Unknown Book XYZ",
        ]
    )
    ids, stats = matcher.match_titles(titles)
    print("ids:", ids.tolist())
    print("stats:", stats)
    assert ids.iloc[0] == 1
    assert ids.iloc[1] == 2
    assert pd.isna(ids.iloc[2])
    print("OK")


if __name__ == "__main__":
    main()

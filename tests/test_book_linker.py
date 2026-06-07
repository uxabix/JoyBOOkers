import pandas as pd

from bookrec.resolution.book_linker import link_books_to_catalog
from bookrec.text_normalization import add_match_keys


def test_isbn_and_match_key_linking():
    catalog = add_match_keys(
        pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Dune", "Foundation"],
                "authors": ["Frank Herbert", "Isaac Asimov"],
                "isbn13": ["9780441172719", "9780553293357"],
            }
        ),
        title_col="name",
        author_col="authors",
    )
    external = add_match_keys(
        pd.DataFrame(
            {
                "source_book_id": ["a", "b"],
                "title": ["Dune", "Unknown Title"],
                "authors": ["Frank Herbert", "Someone"],
                "isbn13": ["9780441172719", None],
            }
        )
    )
    links, stats = link_books_to_catalog(
        external, catalog, source_name="ds2_goodreads_100k", fuzzy_threshold=0
    )
    assert stats["match_isbn"] >= 1
    assert 1 in links["canonical_book_id"].values

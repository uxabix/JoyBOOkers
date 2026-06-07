import pandas as pd

from bookrec.features.content import build_content_features
from bookrec.text_normalization import add_match_keys


def test_content_uses_ds2_only_not_ds1(tmp_path):
    ds2 = add_match_keys(
        pd.DataFrame(
            {
                "source_book_id": ["1", "2"],
                "title": ["Dune", "Foundation"],
                "authors": ["Frank Herbert", "Isaac Asimov"],
                "description": ["Sci-fi epic", "Galactic empire"],
                "genres_list": [["Science Fiction"], ["Science Fiction"]],
            }
        )
    )
    report = build_content_features(ds2, None, tmp_path, max_text_features=100)
    assert "error" not in report
    assert report["ds1_excluded"] is True
    assert report["n_books"] == 2
    assert report["matrix_format"] == "scipy.sparse.csr"


def test_ds3_enriches_ds2_by_match_key(tmp_path):
    ds2 = add_match_keys(
        pd.DataFrame(
            {
                "source_book_id": ["1"],
                "title": ["Dune"],
                "authors": ["Frank Herbert"],
                "description": [""],
                "genres_list": [["Sci-Fi"]],
            }
        )
    )
    ds3 = add_match_keys(
        pd.DataFrame(
            {
                "source_book_id": ["x"],
                "title": ["Dune"],
                "authors": ["Frank Herbert"],
                "tags_list": [["desert", "politics"]],
                "characters_list": [["Paul Atreides"]],
                "genres_list": [[]],
            }
        )
    )
    report = build_content_features(ds2, ds3, tmp_path, max_text_features=100)
    assert report["merge"]["ds3_enriched"] == 1

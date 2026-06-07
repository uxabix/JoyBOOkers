from bookrec.text_normalization import (
    normalize_author_for_match,
    normalize_author_primary,
    normalize_review_text,
    normalize_title_core,
)


def test_normalize_author_pipe_delimited():
    assert normalize_author_primary("Tolkien, J.R.R.|Other") == normalize_author_for_match("Tolkien, J.R.R.")


def test_normalize_review_strips_html():
    assert normalize_review_text("<b>Great</b> book") == "Great book"


def test_title_core_strips_series():
    t = "Dune (Dune Chronicles, #1)"
    assert "dune" in normalize_title_core(t)

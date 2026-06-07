import pandas as pd

from bookrec.splits import split_interactions_per_user, split_nlp_reviews


def test_cf_split_preserves_all_rows():
    inter = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "book_id": [10, 11, 12, 10, 11],
            "rating": [5, 4, 3, 2, 5],
        }
    )
    train, test, report = split_interactions_per_user(inter, test_ratio=0.2, random_state=0)
    assert len(train) + len(test) == len(inter)
    assert report["train_rows"] + report["test_rows"] == len(inter)


def test_nlp_stratified_split():
    reviews = pd.DataFrame(
        {
            "review_text_clean": [f"review text number {i} long enough" for i in range(40)],
            "sentiment_label": [1] * 20 + [0] * 20,
        }
    )
    train, val, test, report = split_nlp_reviews(reviews, random_state=42)
    assert len(train) + len(val) + len(test) == len(reviews)
    assert (train["sentiment_label"] == 1).sum() > 0
    assert (train["sentiment_label"] == 0).sum() > 0

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["figure.dpi"] = 120


def save_rating_distribution(interactions: pd.DataFrame, out_path: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots()
    order = [1, 2, 3, 4, 5]
    vc = interactions["rating"].value_counts().reindex(order, fill_value=0)
    sns.barplot(x=vc.index.astype(int), y=vc.values, ax=ax, color="steelblue")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of user ratings")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_top_books(interactions: pd.DataFrame, books: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    _setup_style()
    counts = interactions.groupby("book_id").size().sort_values(ascending=False).head(top_n)
    book_names = books.set_index("id")["name"] if "name" in books.columns else None
    labels = []
    for bid in counts.index:
        if book_names is not None and bid in book_names.index:
            t = str(book_names.loc[bid])
            labels.append(t[:50] + ("…" if len(t) > 50 else ""))
        else:
            labels.append(str(bid))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=counts.values, y=labels, ax=ax, orient="h", color="darkslateblue")
    ax.set_xlabel("Number of ratings")
    ax.set_title(f"Top {top_n} books by interaction count")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_top_users(interactions: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    _setup_style()
    counts = interactions.groupby("user_id").size().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="seagreen")
    ax.set_xlabel("user_id")
    ax.set_ylabel("Ratings given")
    ax.set_title(f"Top {top_n} most active users")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_language_distribution(books: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    if "language" not in books.columns:
        return
    _setup_style()
    lang = books["language"].dropna().astype(str)
    lang = lang[lang.str.len() > 0]
    vc = lang.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h", color="coral")
    ax.set_xlabel("Number of books")
    ax.set_title(f"Top {top_n} languages in catalog")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_books_per_year(books: pd.DataFrame, out_path: Path) -> None:
    if "publishyear" not in books.columns:
        return
    _setup_style()
    y = pd.to_numeric(books["publishyear"], errors="coerce").dropna().astype(int)
    y = y[(y >= 1800) & (y <= 2030)]
    vc = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    vc.plot(ax=ax, kind="line", color="steelblue")
    ax.set_xlabel("Publish year")
    ax.set_ylabel("Number of books")
    ax.set_title("Books in dataset by publication year")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_top_publishers(books: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    if "publisher" not in books.columns:
        return
    _setup_style()
    pub = books["publisher"].dropna().astype(str)
    pub = pub[pub.str.len() > 0]
    vc = pub.value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h", color="indianred")
    ax.set_xlabel("Number of books")
    ax.set_title(f"Top {top_n} publishers")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_avg_book_rating_hist(books: pd.DataFrame, out_path: Path) -> None:
    if "rating" not in books.columns:
        return
    _setup_style()
    r = pd.to_numeric(books["rating"], errors="coerce").dropna()
    r = r[(r >= 0) & (r <= 5)]
    fig, ax = plt.subplots()
    sns.histplot(r, bins=40, kde=True, ax=ax, color="teal")
    ax.set_xlabel("Average book rating (catalog)")
    ax.set_title("Distribution of average ratings in book.csv")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

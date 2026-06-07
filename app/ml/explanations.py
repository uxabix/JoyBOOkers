"""Human-readable recommendation explanations from signal breakdown."""

from __future__ import annotations

from app.ml.user_profile import UserProfile


def build_explanations(
    profile: UserProfile,
    breakdown: dict[str, float],
    *,
    book_genre: str | None = None,
    threshold: float = 0.45,
) -> list[str]:
    """Return up to 3 short explanation strings for a recommended book."""
    ranked = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
    top_key = ranked[0][0] if ranked else ""
    out: list[str] = []

    if breakdown.get("content", 0) >= threshold and profile.rated_books:
        genres = list(profile.genre_weights.keys())[:2]
        if genres:
            out.append(f"Similar to books you rated ({', '.join(genres)})")
        else:
            out.append("Similar to books you rated (content match)")

    if breakdown.get("cf", 0) >= threshold and profile.cf_available:
        out.append("Users with similar tastes rated this highly (collaborative filtering)")

    if breakdown.get("cluster", 0) >= threshold:
        label = profile.cluster_label or f"cluster {profile.cluster_id}"
        out.append(f"Popular among readers like you ({label})")

    if breakdown.get("pop", 0) >= threshold and top_key == "pop":
        out.append("Frequently rated in the catalog")

    if breakdown.get("genre", 0) >= threshold:
        if profile.rated_books and profile.genre_weights:
            top = max(profile.genre_weights, key=profile.genre_weights.get)
            out.append(f"Matches your preference for {top}")
        elif book_genre:
            out.append(f"Fits popular genres in your reader profile ({book_genre.split(',')[0].strip()})")
        else:
            out.append("Aligns with genre trends in your reader cluster")

    if not out and breakdown.get("pop", 0) > 0:
        out.append("Recommended based on catalog popularity and your reader profile")

    return out[:3]

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
            out.append(f"Podobne do książek, które oceniłeś/aś ({', '.join(genres)})")
        else:
            out.append("Podobne do książek, które oceniłeś/aś (dopasowanie treści)")

    if breakdown.get("cf", 0) >= threshold and profile.cf_available:
        out.append("Użytkownicy o podobnych gustach wysoko ocenili tę pozycję (filtrowanie współpracujące)")

    if breakdown.get("cluster", 0) >= threshold:
        label = profile.cluster_label or f"klaster {profile.cluster_id}"
        out.append(f"Popularne wśród czytelników podobnych do Ciebie ({label})")

    if breakdown.get("pop", 0) >= threshold and top_key == "pop":
        out.append("Często oceniana w katalogu")

    if breakdown.get("genre", 0) >= threshold:
        if profile.rated_books and profile.genre_weights:
            top = max(profile.genre_weights, key=profile.genre_weights.get)
            out.append(f"Pasuje do Twojej preferencji: {top}")
        elif book_genre:
            out.append(
                f"Pasuje do popularnych gatunków w Twoim profilu ({book_genre.split(',')[0].strip()})"
            )
        else:
            out.append("Zgodne z trendami gatunkowymi w Twoim klastrze czytelniczym")

    if not out and breakdown.get("pop", 0) > 0:
        out.append("Rekomendacja na podstawie popularności w katalogu i Twojego profilu czytelnika")

    return out[:3]

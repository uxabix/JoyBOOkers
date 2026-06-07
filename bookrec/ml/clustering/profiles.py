"""Rich cluster profiles and data-driven descriptions for K-Means user segments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STAR_LABELS = ("1", "2", "3", "4", "5")


def _rating_distribution(series: pd.Series) -> dict[str, float]:
    """Share of each star rating (1–5) within a group."""
    counts = series.astype(int).value_counts()
    total = float(counts.sum()) or 1.0
    return {star: round(float(counts.get(int(star), 0)) / total, 4) for star in STAR_LABELS}


def _activity_mix(activity: pd.Series) -> dict[str, float]:
    counts = activity.value_counts()
    total = float(len(activity)) or 1.0
    return {
        level: round(float(counts.get(level, 0)) / total, 4)
        for level in ("low", "medium", "high")
    }


def _baseline_from_profiles(clusters: dict[str, dict[str, Any]]) -> dict[str, float]:
    if not clusters:
        return {}
    n_users = sum(int(c.get("n_users", 0)) for c in clusters.values()) or 1
    return {
        "n_ratings_mean": sum(c["n_ratings_mean"] * c["n_users"] for c in clusters.values()) / n_users,
        "mean_rating": sum(c["mean_rating"] * c["n_users"] for c in clusters.values()) / n_users,
        "std_rating": sum(c["std_rating"] * c["n_users"] for c in clusters.values()) / n_users,
    }


def describe_cluster_pl(profile: dict[str, Any], baseline: dict[str, float]) -> tuple[str, str]:
    """Build short title and full Polish description from cluster metrics."""
    n_mean = float(profile["n_ratings_mean"])
    m_rating = float(profile["mean_rating"])
    std_r = float(profile["std_rating"])
    base_n = float(baseline.get("n_ratings_mean", n_mean) or n_mean)
    base_m = float(baseline.get("mean_rating", m_rating) or m_rating)

    dist = profile.get("rating_distribution_pct") or {}
    top_star = max(STAR_LABELS, key=lambda s: float(dist.get(s, 0)))
    top_pct = float(dist.get(top_star, 0)) * 100

    activity = profile.get("activity_mix") or {}
    dominant_activity = max(("low", "medium", "high"), key=lambda k: float(activity.get(k, 0)))

    # Short title (badge / cluster_label) — activity tier + mean volume vs global baseline
    if dominant_activity == "high" or n_mean >= base_n * 1.25:
        title = "Aktywni czytelnicy"
    elif dominant_activity == "medium" or (base_n * 0.75 < n_mean < base_n * 1.25):
        title = "Umiarkowana aktywność"
    else:
        title = "Okazjonalni oceniający"

    if m_rating >= base_m + 0.15:
        title += " — hojni w ocenach"
    elif m_rating <= base_m - 0.15:
        title += " — bardziej krytyczni"

    parts: list[str] = []

    share_pct = float(profile.get("share_of_users_pct", 0))
    parts.append(f"{int(profile.get('n_users', 0))} użytkowników ({share_pct:.1f}% próby)")

    if n_mean >= base_n * 1.25:
        parts.append(f"wysoka aktywność: średnio {n_mean:.0f} ocen na użytkownika")
    elif n_mean <= base_n * 0.75:
        parts.append(f"niska aktywność: średnio {n_mean:.1f} ocen na użytkownika")
    else:
        parts.append(f"średnio {n_mean:.1f} ocen na użytkownika")

    parts.append(f"średnia ocena {m_rating:.2f} (globalnie {base_m:.2f})")

    if std_r >= 0.85:
        parts.append(f"duży rozrzut ocen (std {std_r:.2f})")
    elif std_r <= 0.72:
        parts.append(f"stabilne oceny (std {std_r:.2f})")

    activity_pl = {"low": "niska", "medium": "średnia", "high": "wysoka"}
    parts.append(
        f"dominująca aktywność: {activity_pl[dominant_activity]} "
        f"({activity.get(dominant_activity, 0) * 100:.0f}% użytkowników)"
    )

    parts.append(
        f"rozkład ocen: najczęściej {top_star}★ ({top_pct:.0f}% wszystkich ocen w klastrze)"
    )

    generous = float(profile.get("generous_share_pct", 0))
    critical = float(profile.get("critical_share_pct", 0))
    if generous >= 55:
        parts.append(f"{generous:.0f}% ocen to 4–5★")
    if critical >= 15:
        parts.append(f"{critical:.0f}% ocen to 1–2★")

    return title, "; ".join(parts) + "."


def build_cluster_profiles_detail(
    *,
    user_features: pd.DataFrame,
    cluster_ids: pd.Series,
    interactions: pd.DataFrame | None = None,
    cluster_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Aggregate per-cluster metrics including star-rating distribution."""
    merged_users = user_features.copy()
    merged_users["cluster_id"] = cluster_ids.astype(int).values

    n_total_users = len(merged_users)
    clusters: dict[str, dict[str, Any]] = {}

    for cluster_id, group in merged_users.groupby("cluster_id"):
        cid = str(int(cluster_id))
        n_users = int(len(group))
        profile: dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "n_users": n_users,
            "share_of_users_pct": round(100.0 * n_users / n_total_users, 2) if n_total_users else 0.0,
            "n_ratings_mean": round(float(group["n_ratings"].mean()), 3),
            "n_ratings_median": round(float(group["n_ratings"].median()), 3),
            "mean_rating": round(float(group["mean_rating"].mean()), 3),
            "std_rating": round(float(group["std_rating"].mean()), 3),
            "rating_range_mean": round(float(group["rating_range"].mean()), 3)
            if "rating_range" in group.columns
            else None,
            "min_rating_mean": round(float(group["min_rating"].mean()), 3)
            if "min_rating" in group.columns
            else None,
            "max_rating_mean": round(float(group["max_rating"].mean()), 3)
            if "max_rating" in group.columns
            else None,
        }

        if "activity_level" in group.columns:
            profile["activity_mix"] = _activity_mix(group["activity_level"])

        if interactions is not None and not interactions.empty:
            inter = interactions.copy()
            inter["user_id"] = inter["user_id"].astype(str)
            user_ids = group["user_id"].astype(str)
            cluster_inter = inter[inter["user_id"].isin(user_ids)]
            if not cluster_inter.empty:
                profile["rating_distribution_pct"] = _rating_distribution(cluster_inter["rating"])
                profile["n_interactions"] = int(len(cluster_inter))
                ratings = cluster_inter["rating"].astype(float)
                profile["generous_share_pct"] = round(float((ratings >= 4).mean()) * 100, 2)
                profile["critical_share_pct"] = round(float((ratings <= 2).mean()) * 100, 2)
                profile["neutral_share_pct"] = round(float((ratings == 3).mean()) * 100, 2)
            else:
                profile["rating_distribution_pct"] = {s: 0.0 for s in STAR_LABELS}
        else:
            profile["rating_distribution_pct"] = {s: 0.0 for s in STAR_LABELS}

        clusters[cid] = profile

    baseline = _baseline_from_profiles(clusters)
    descriptions: dict[str, dict[str, str]] = {}
    for cid, profile in clusters.items():
        title, description = describe_cluster_pl(profile, baseline)
        profile["title"] = title
        profile["description"] = description
        descriptions[cid] = {"title": title, "description": description}

    # Legacy shape for charts expecting cluster_profiles_mean
    profiles_mean = {
        "n_ratings": {cid: clusters[cid]["n_ratings_mean"] for cid in clusters},
        "mean_rating": {cid: clusters[cid]["mean_rating"] for cid in clusters},
        "std_rating": {cid: clusters[cid]["std_rating"] for cid in clusters},
    }

    return {
        "baseline": baseline,
        "clusters": clusters,
        "cluster_descriptions": descriptions,
        "cluster_profiles_mean": profiles_mean,
        "cluster_sizes": cluster_sizes or {cid: clusters[cid]["n_users"] for cid in clusters},
    }

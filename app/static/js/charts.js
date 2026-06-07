/**
 * Chart.js helpers for analytics and clustering dashboards.
 * Expects window.CHART_DATA set by the template.
 */
(function () {
  "use strict";

  const palette = [
    "#0d6efd",
    "#198754",
    "#fd7e14",
    "#6610f2",
    "#dc3545",
    "#0dcaf0",
    "#6f42c1",
  ];

  function clusterIdFromLabel(label) {
    const match = String(label).match(/(\d+)/);
    return match ? parseInt(match[1], 10) : -1;
  }

  function colorsForClusters(labels, highlightId) {
    return labels.map(function (label, i) {
      const cid = clusterIdFromLabel(label);
      const base = palette[cid] || palette[i % palette.length];
      if (highlightId === undefined || highlightId === null) {
        return base;
      }
      return cid === highlightId ? base : base + "44";
    });
  }

  function borderForClusters(labels, highlightId) {
    return labels.map(function (label, i) {
      const cid = clusterIdFromLabel(label);
      if (highlightId === undefined || highlightId === null) {
        return "transparent";
      }
      return cid === highlightId ? "#212529" : "transparent";
    });
  }

  function borderWidthForClusters(labels, highlightId) {
    return labels.map(function (label) {
      const cid = clusterIdFromLabel(label);
      if (highlightId === undefined || highlightId === null) {
        return 0;
      }
      return cid === highlightId ? 3 : 0;
    });
  }

  function renderBar(canvasId, labels, values, label, highlightId) {
    const el = document.getElementById(canvasId);
    if (!el || typeof Chart === "undefined") return;
    const ctx = el.getContext("2d");
    new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: label,
            data: values,
            backgroundColor: colorsForClusters(labels, highlightId),
            borderColor: borderForClusters(labels, highlightId),
            borderWidth: borderWidthForClusters(labels, highlightId),
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } },
      },
    });
  }

  function renderDoughnut(canvasId, labels, values, highlightId) {
    const el = document.getElementById(canvasId);
    if (!el || typeof Chart === "undefined") return;
    const ctx = el.getContext("2d");
    new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [
          {
            data: values,
            backgroundColor: colorsForClusters(labels, highlightId),
            borderColor: borderForClusters(labels, highlightId),
            borderWidth: borderWidthForClusters(labels, highlightId),
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
      },
    });
  }

  function renderLine(canvasId, labels, values, label) {
    const el = document.getElementById(canvasId);
    if (!el || typeof Chart === "undefined") return;
    const ctx = el.getContext("2d");
    new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: label,
            data: values,
            borderColor: palette[0],
            backgroundColor: "rgba(13, 110, 253, 0.1)",
            fill: true,
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 1 } },
      },
    });
  }

  function initAnalytics() {
    const data = window.CHART_DATA;
    if (!data) return;
    if (data.ml_comparison) {
      renderBar(
        "chart-ml-comparison",
        data.ml_comparison.labels,
        data.ml_comparison.values,
        "ML metrics"
      );
    }
    if (data.dataset_volumes) {
      renderBar(
        "chart-dataset-volumes",
        data.dataset_volumes.labels,
        data.dataset_volumes.values,
        "Row counts"
      );
    }
  }

  function initClustering() {
    const data = window.CHART_DATA;
    if (!data) return;
    const highlightId = data.highlight_cluster_id;
    if (data.cluster_sizes) {
      renderDoughnut(
        "chart-cluster-sizes",
        data.cluster_sizes.labels,
        data.cluster_sizes.values,
        highlightId
      );
    }
    if (data.silhouette_by_k) {
      renderLine(
        "chart-silhouette-k",
        data.silhouette_by_k.labels,
        data.silhouette_by_k.values,
        "Silhouette score"
      );
    }
    if (data.profile_ratings) {
      renderBar(
        "chart-profile-ratings",
        data.profile_ratings.labels,
        data.profile_ratings.mean_ratings,
        "Mean rating",
        highlightId
      );
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("chart-ml-comparison")) initAnalytics();
    if (document.getElementById("chart-cluster-sizes")) initClustering();
  });
})();

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
        "Metryki ML"
      );
    }
    if (data.dataset_volumes) {
      renderBar(
        "chart-dataset-volumes",
        data.dataset_volumes.labels,
        data.dataset_volumes.values,
        "Liczba wierszy"
      );
    }
  }

  var clusterPcaState = {
    chart: null,
    canvasId: "chart-pca-scatter",
    pcaData: null,
    highlightId: null,
    baseHighlights: [],
  };

  function highlightDatasets(highlightPoints) {
    return (highlightPoints || []).map(function (pt) {
      var isUser = pt.kind === "user";
      var clusterColor = palette[parseInt(pt.cluster_id, 10)] || "#ffc107";
      return {
        label: pt.label || (isUser ? "Ty" : "Symulacja"),
        data: [{ x: pt.x, y: pt.y }],
        pointRadius: isUser ? 16 : 13,
        pointHoverRadius: 20,
        pointStyle: isUser ? "star" : "triangle",
        backgroundColor: isUser ? "#ffc107" : "#ff6b6b",
        borderColor: isUser ? "#212529" : clusterColor,
        borderWidth: 3,
        order: -1,
      };
    });
  }

  function renderScatterPca(canvasId, pcaData, highlightId, highlightPoints) {
    const el = document.getElementById(canvasId);
    if (!el || typeof Chart === "undefined" || !pcaData || !pcaData.points_by_cluster) {
      return;
    }
    if (clusterPcaState.chart) {
      clusterPcaState.chart.destroy();
      clusterPcaState.chart = null;
    }
    const ctx = el.getContext("2d");
    const clusters = Object.keys(pcaData.points_by_cluster).sort(function (a, b) {
      return parseInt(a, 10) - parseInt(b, 10);
    });
    const datasets = clusters.map(function (cid, i) {
      const label = "Klaster " + cid;
      const base = palette[parseInt(cid, 10)] || palette[i % palette.length];
      const dimmed = highlightId !== undefined && highlightId !== null && parseInt(cid, 10) !== highlightId;
      return {
        label: label,
        data: pcaData.points_by_cluster[cid],
        pointRadius: dimmed ? 2 : 3.5,
        pointHoverRadius: 5,
        backgroundColor: dimmed ? base + "33" : base + "aa",
        borderColor: dimmed ? base + "55" : base,
        borderWidth: parseInt(cid, 10) === highlightId ? 2 : 0,
        order: 1,
      };
    });
    datasets.push.apply(datasets, highlightDatasets(highlightPoints));

    clusterPcaState.chart = new Chart(ctx, {
      type: "scatter",
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
          title: {
            display: !!(pcaData.explained_variance_pct && pcaData.explained_variance_pct.length),
            text:
              "PCA: PC1 " +
              (pcaData.explained_variance_pct[0] || 0) +
              "%, PC2 " +
              (pcaData.explained_variance_pct[1] || 0) +
              "% wariancji",
            font: { size: 12 },
          },
        },
        scales: {
          x: { title: { display: true, text: "PC1" } },
          y: { title: { display: true, text: "PC2" } },
        },
      },
    });
    clusterPcaState.canvasId = canvasId;
    clusterPcaState.pcaData = pcaData;
    clusterPcaState.highlightId = highlightId;
    clusterPcaState.baseHighlights = (highlightPoints || []).filter(function (p) {
      return p.kind === "user";
    });
  }

  window.setPcaPreviewHighlight = function (previewPoint) {
    var merged = clusterPcaState.baseHighlights.slice();
    if (previewPoint) {
      merged.push(previewPoint);
    }
    renderScatterPca(
      clusterPcaState.canvasId,
      clusterPcaState.pcaData,
      clusterPcaState.highlightId,
      merged
    );
  };

  function renderGroupedRatingsCount(canvasId, histData, highlightId) {
    const el = document.getElementById(canvasId);
    if (!el || typeof Chart === "undefined" || !histData || !histData.by_cluster) {
      return;
    }
    const ctx = el.getContext("2d");
    const labels = histData.bin_labels || [];
    const clusterIds = Object.keys(histData.by_cluster).sort(function (a, b) {
      return parseInt(a, 10) - parseInt(b, 10);
    });
    const datasets = clusterIds.map(function (cid, i) {
      const base = palette[parseInt(cid, 10)] || palette[i % palette.length];
      const dimmed = highlightId !== undefined && highlightId !== null && parseInt(cid, 10) !== highlightId;
      return {
        label: "Klaster " + cid,
        data: histData.by_cluster[cid],
        backgroundColor: dimmed ? base + "44" : base,
        borderColor: base,
        borderWidth: 1,
      };
    });
    new Chart(ctx, {
      type: "bar",
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
          title: {
            display: true,
            text: "Liczba użytkowników wg liczby wystawionych ocen",
            font: { size: 12 },
          },
        },
        scales: {
          x: {
            stacked: false,
            title: { display: true, text: "Liczba ocen na użytkownika" },
          },
          y: {
            beginAtZero: true,
            title: { display: true, text: "Użytkownicy" },
          },
        },
      },
    });
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
        "Współczynnik silhouette"
      );
    }
    if (data.profile_ratings) {
      renderBar(
        "chart-profile-ratings",
        data.profile_ratings.labels,
        data.profile_ratings.mean_ratings,
        "Średnia ocena",
        highlightId
      );
    }
    if (data.rating_distributions) {
      Object.keys(data.rating_distributions).forEach(function (cid) {
        var block = data.rating_distributions[cid];
        var canvasId = "chart-rating-dist-" + cid;
        if (document.getElementById(canvasId)) {
          renderBar(canvasId, block.labels, block.values, "Oceny %");
        }
      });
    }
    if (data.pca_scatter) {
      renderScatterPca(
        "chart-pca-scatter",
        data.pca_scatter,
        highlightId,
        data.pca_highlights || []
      );
    }
    if (data.n_ratings_histogram) {
      renderGroupedRatingsCount("chart-n-ratings-hist", data.n_ratings_histogram, highlightId);
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    if (document.getElementById("chart-ml-comparison")) initAnalytics();
    if (document.getElementById("chart-cluster-sizes")) initClustering();
  });
})();

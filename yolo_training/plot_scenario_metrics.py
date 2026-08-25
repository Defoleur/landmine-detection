import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS_PATH = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_scenario_eval_ultimate/metrics.json"
PLOTS_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/datasets/dataset_scenario_eval_ultimate/plots"
CLASS_NAMES = [
    "mine_disc",
    "mine_rectangular_elongated",
    "mine_square",
    "mine_rectangular",
    "mine_cylinder_vert",
    "mine_directional_standing",
    "mine_round_directional_standing",
]

SCENARIOS_ORDER = [
    "spring_day", "spring_morning", "spring_evening",
    "summer_day", "summer_morning", "summer_evening",
    "autumn_day", "autumn_morning", "autumn_evening",
    "winter_day", "winter_morning", "winter_evening",
]


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def plot_mAP(metrics, scenarios, x, bar_w):
    fig, ax = plt.subplots(figsize=(14, 6))
    map50 = [metrics[s]["mAP50"] for s in scenarios]
    map5095 = [metrics[s]["mAP50-95"] for s in scenarios]
    bars1 = ax.bar(x - bar_w / 2, map50, bar_w, label="mAP50", color="#4C72B0")
    bars2 = ax.bar(x + bar_w / 2, map5095, bar_w, label="mAP50-95", color="#DD8452")
    ax.set_ylabel("mAP")
    ax.set_title("Detection mAP per Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mAP_per_scenario.png"), dpi=150)
    plt.close()
    print("  saved mAP_per_scenario.png")


def plot_precision_recall(metrics, scenarios, x, bar_w):
    fig, ax = plt.subplots(figsize=(14, 6))
    precs = [metrics[s]["precision"] for s in scenarios]
    recs = [metrics[s]["recall"] for s in scenarios]
    bars1 = ax.bar(x - bar_w / 2, precs, bar_w, label="Precision", color="#55A868")
    bars2 = ax.bar(x + bar_w / 2, recs, bar_w, label="Recall", color="#C44E52")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall per Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_per_scenario.png"), dpi=150)
    plt.close()
    print("  saved precision_recall_per_scenario.png")


def plot_heatmap(metrics, scenarios):
    matrix = np.zeros((len(CLASS_NAMES), len(scenarios)))
    for j, sc in enumerate(scenarios):
        pca = metrics[sc]["per_class_ap"]
        for i, cls in enumerate(CLASS_NAMES):
            matrix[i, j] = pca.get(cls, 0)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("Per-Class AP across Scenarios")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(scenarios)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    plt.colorbar(im, ax=ax, label="AP")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_class_heatmap.png"), dpi=150)
    plt.close()
    print("  saved per_class_heatmap.png")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics = load_metrics()
    scenarios = [s for s in SCENARIOS_ORDER if s in metrics]
    x = np.arange(len(scenarios))
    bar_w = 0.35

    print("Generating plots...")
    plot_mAP(metrics, scenarios, x, bar_w)
    plot_precision_recall(metrics, scenarios, x, bar_w)
    plot_heatmap(metrics, scenarios)
    print("Done!")


if __name__ == "__main__":
    main()

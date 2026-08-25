import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "/media/geekscode/Developer1/Projects/gd-autopilot/yolo_training/trained_models_project/landmine_multimodal_oblique_7cls_e150_grouped"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

LOSS_PAIRS = [
    ("train/box_loss", "val/box_loss", "Box Loss"),
    ("train/cls_loss", "val/cls_loss", "Class Loss"),
    ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
]

METRIC_PAIRS = [
    ("metrics/mAP50(B)", "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50-95"),
]

SINGLE_METRICS = [
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)", "Recall"),
]


def plot_pair(df, col, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df[col], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))

    # loss curves (train + val on same plot)
    for train_col, val_col, title in LOSS_PAIRS:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["epoch"], df[train_col], label="Train", linewidth=1.5)
        ax.plot(df["epoch"], df[val_col], label="Val", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = title.lower().replace(" ", "_") + ".png"
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {fname}")

    # mAP metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", linewidth=1.5)
    ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, "map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved map.png")

    # precision & recall on one plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linewidth=1.5)
    ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Precision & Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, "precision_recall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved precision_recall.png")


if __name__ == "__main__":
    main()

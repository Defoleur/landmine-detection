"""Streaming binary segmentation metrics for validation."""

import torch


class SegmentationMetrics:
    def __init__(self, bins: int = 1000):
        self.bins = bins
        self.tp = self.fp = self.fn = self.tn = self.images = 0
        self.probability_hist = torch.zeros(bins, dtype=torch.float64)
        self.positive_hist = torch.zeros(bins, dtype=torch.float64)

    def update(self, logits, target, mask, threshold=0.5):
        valid = mask.bool()
        probabilities = torch.sigmoid(logits.detach())[valid].cpu()
        labels = target.detach().bool()[valid].cpu()
        predicted = probabilities >= threshold
        self.tp += int((predicted & labels).sum())
        self.fp += int((predicted & ~labels).sum())
        self.fn += int((~predicted & labels).sum())
        self.tn += int((~predicted & ~labels).sum())
        self.images += logits.shape[0]
        indices = (probabilities.clamp(0, 1) * (self.bins - 1)).long()
        self.probability_hist.scatter_add_(0, indices, torch.ones_like(probabilities, dtype=torch.float64))
        self.positive_hist.scatter_add_(0, indices, labels.to(torch.float64))

    def compute(self):
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        specificity = self.tn / max(self.tn + self.fp, 1)
        dice = 2 * self.tp / max(2 * self.tp + self.fp + self.fn, 1)
        iou = self.tp / max(self.tp + self.fp + self.fn, 1)
        false_positives_per_image = self.fp / max(self.images, 1)
        total_positives = self.positive_hist.sum().item()
        if total_positives == 0:
            pr_auc = 0.0
        else:
            predictions = torch.cumsum(self.probability_hist.flip(0), 0)
            positives = torch.cumsum(self.positive_hist.flip(0), 0)
            curve_precision = positives / predictions.clamp_min(1)
            curve_recall = positives / total_positives
            pr_auc = float(torch.sum((curve_recall[1:] - curve_recall[:-1]) * curve_precision[1:]))
        return {"precision": precision, "recall": recall, "specificity": specificity,
                "dice": dice, "iou": iou, "pr_auc": pr_auc,
                "false_positives_per_image": false_positives_per_image}

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.insert(0, "src")

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from data import prepare_train_val_test, build_train_val_dataloaders, build_test_dataloader, LABEL_COLS
from dataset import CustomDataset
from transforms import build_transforms
from models import build_resnet50, build_efficientnet_b1
from train_utils import get_device
from evaluate import find_threshold, compute_auroc_auprc

device = get_device()
best_alpha = 0.5
target_recall = 0.85

CSV_PATH = "label_chexpert.csv"
root = "images_224"

df = pd.read_csv(CSV_PATH)

(
    train_df, val_df, test_df,
    train_path, val_path, test_path,
    train_target, val_target, test_target,
    train_mask, val_mask, test_mask,
    pos_weight
) = prepare_train_val_test(df, root)

_, val_transform = build_transforms()

val_dataset = CustomDataset(
    path=val_path,
    target=val_target,
    mask=val_mask,
    transform=val_transform,
)

_, val_loader = build_train_val_dataloaders(
    train_dataset=val_dataset,
    val_dataset=val_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
)

test_dataset = CustomDataset(
    path=test_path,
    target=test_target,
    mask=test_mask,
    transform=val_transform,
)

test_loader = build_test_dataloader(
    test_dataset=test_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
)

num_classes = val_target.shape[1]

resnet_model = build_resnet50(
    unfreeze_from=None,
    num_classes=num_classes
).to(device)

resnet_model.load_state_dict(
    torch.load("best_resnet.pth", map_location=device)
)

efficient_model = build_efficientnet_b1(
    unfreeze_last_n_blocks=0,
    num_classes=num_classes
).to(device)

efficient_model.load_state_dict(
    torch.load("best_efficientnet.pth", map_location=device)
)

resnet_model.eval()
efficient_model.eval()

val_probs = []
val_labels = []
val_masks = []

with torch.no_grad():
    for images, labels, masks in tqdm(val_loader, desc="Collect Val Probs"):
        images = images.to(device)
        labels = labels.to(device)

        probs1 = torch.sigmoid(resnet_model(images))
        probs2 = torch.sigmoid(efficient_model(images))
        probs = best_alpha * probs1 + (1 - best_alpha) * probs2

        val_probs.append(probs.cpu())
        val_labels.append(labels.cpu())
        val_masks.append(masks.cpu())

val_probs = torch.cat(val_probs).numpy()
val_targets = torch.cat(val_labels).numpy()
val_masks = torch.cat(val_masks).numpy()

thresholds_dict = find_threshold(
    probs=val_probs,
    targets=val_targets,
    masks=val_masks,
    target_recall=target_recall
)

all_test_probs = []
all_test_labels = []
all_test_masks = []

with torch.no_grad():
    for images, labels, masks in tqdm(test_loader, desc="Test Ensemble"):
        images = images.to(device)
        labels = labels.to(device)
        masks  = masks.to(device)

        probs1 = torch.sigmoid(resnet_model(images))
        probs2 = torch.sigmoid(efficient_model(images))
        probs  = best_alpha * probs1 + (1 - best_alpha) * probs2

        all_test_probs.append(probs.cpu())
        all_test_labels.append(labels.cpu())
        all_test_masks.append(masks.cpu())

test_probs  = torch.cat(all_test_probs).numpy()
test_labels = torch.cat(all_test_labels).numpy()
test_masks  = torch.cat(all_test_masks).numpy()

metrics = compute_auroc_auprc(
    probs=test_probs,
    labels=test_labels,
    masks=test_masks,
    class_names=LABEL_COLS
)

print("\n===== Test Results (Threshold-independent) =====")
print(f"Macro AUROC : {metrics['macro_auroc']:.4f}")
print(f"Macro AUPRC : {metrics['macro_auprc']:.4f}")

print("\nClass-wise AUPRC:")
for cls, score in metrics["auprc_per_class"].items():
    print(f"{cls}: {score:.4f}")

correct = 0.0
total = 0.0

for c, thr in thresholds_dict.items():
    preds = (test_probs[:, c] >= thr).astype(float)
    mask  = test_masks[:, c] == 1

    correct += ((preds == test_labels[:, c]) * mask).sum()
    total   += mask.sum()

acc = correct / max(total, 1.0)

print("\n===== Test Results (Class-wise Threshold) =====")
print(f"Recall constraint : ≥ {target_recall}")
print(f"Accuracy          : {acc:.4f}")


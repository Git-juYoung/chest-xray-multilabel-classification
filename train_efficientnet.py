#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, "src")

import pandas as pd
import torch

from seed import set_seed
from config import e_config
from data import prepare_train_val_test, build_train_val_dataloaders, LABEL_COLS
from dataset import CustomDataset
from transforms import build_transforms
from models import build_efficientnet_b1
from engine import train_one_epoch, validate_one_epoch
from evaluate import compute_auroc_auprc
from train_utils import (
    get_device,
    build_criterion,
    build_optimizer,
    build_scheduler,
)
from early_stopping import EarlyStopping

import wandb

set_seed()

WANDB_PROJECT = "chexpert_multilabel"
WANDB_RUN_NAME = "efficientnet_1차"

run = wandb.init(
project=WANDB_PROJECT,
name=WANDB_RUN_NAME,
config=e_config
)

CSV_PATH = "label_chexpert.csv"
IMAGE_ROOT = "images_224"

df = pd.read_csv(CSV_PATH)

(
    train_df, val_df, _,
    train_path, val_path, _,
    train_target, val_target, _,
    train_mask, val_mask, _,
    pos_weight
) = prepare_train_val_test(
    df=df,
    root=IMAGE_ROOT,
    random_state=42,
    verbose=True
)

train_transform, val_test_transform = build_transforms()

train_dataset = CustomDataset(
    path=train_path,
    target=train_target,
    mask=train_mask,
    transform=train_transform,
)

val_dataset = CustomDataset(
    path=val_path,
    target=val_target,
    mask=val_mask,
    transform=val_test_transform,
)

train_loader, val_loader = build_train_val_dataloaders(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=e_config["batch_size"],
    num_workers=e_config["num_workers"],
    pin_memory=True,
)

device = get_device()

num_classes = train_target.shape[1]

model = build_efficientnet_b1(
    unfreeze_last_n_blocks=e_config["unfreeze_last_n_blocks"],
    num_classes=num_classes,
).to(device)

criterion = build_criterion(
    pos_weight=pos_weight,
    device=device
)

optimizer = build_optimizer(
    model,
    lr=e_config["optimizer"]["lr"],
    weight_decay=e_config["optimizer"]["weight_decay"],
)

scheduler = build_scheduler(
    optimizer,
    mode="min",
    factor=e_config["scheduler"]["factor"],
    patience=e_config["scheduler"]["patience"],
)

early_stopper = EarlyStopping(
    patience=e_config["early_stopping"]["patience"],
    save_path="best_efficientnet.pth",
)

num_epochs = e_config["num_epochs"]

for epoch in range(1, num_epochs + 1):

    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        name="EfficientNet",
        epoch=epoch,
        num_epochs=num_epochs,
    )

    val_loss = validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        name="EfficientNet",
        epoch=epoch,
        num_epochs=num_epochs,
    )

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    
    wandb.log(
        {
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "lr": current_lr,
        },
        step=epoch,
    )
    
    print(
        f"[Epoch {epoch}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if epoch % 5 == 0:
        model.eval()

        all_probs = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)

                logits = model(images)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_masks.append(masks.cpu())

        metrics = compute_auroc_auprc(
            probs=torch.cat(all_probs).numpy(),
            labels=torch.cat(all_labels).numpy(),
            masks=torch.cat(all_masks).numpy(),
            class_names=LABEL_COLS
        )

        wandb.log(
            {
            "val/macro_auroc": metrics["macro_auroc"],
            "val/macro_auprc": metrics["macro_auprc"],
            },
            step=epoch,
        )

        print(
            f" [Val Metrics @ Epoch {epoch}] "
            f"AUROC: {metrics['macro_auroc']:.4f} | "
            f"AUPRC: {metrics['macro_auprc']:.4f}"
        )

    if early_stopper.step(val_loss, model):
        print("Early stopping triggered.")
        break

print("\nLoading best model for final evaluation...")

model.load_state_dict(torch.load("best_efficientnet.pth"))
model.to(device)
model.eval()

all_probs = []
all_labels = []
all_masks = []

with torch.no_grad():
    for images, labels, masks in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        all_masks.append(masks.cpu())

final_metrics = compute_auroc_auprc(
    probs=torch.cat(all_probs).numpy(),
    labels=torch.cat(all_labels).numpy(),
    masks=torch.cat(all_masks).numpy(),
    class_names=LABEL_COLS
)

print("\n[Best Model - Per Class Performance]")
for name, value in final_metrics["auroc_per_class"].items():
    print(f"{name} AUROC: {value:.4f}")

for name, value in final_metrics["auprc_per_class"].items():
    print(f"{name} AUPRC: {value:.4f}")

for name, value in final_metrics["auroc_per_class"].items():
    wandb.log({f"best/class_auroc/{name}": value})

for name, value in final_metrics["auprc_per_class"].items():
    wandb.log({f"best/class_auprc/{name}": value})

print(
    f"\n[Best Model Validation Performance] "
    f"AUROC: {final_metrics['macro_auroc']:.4f} | "
    f"AUPRC: {final_metrics['macro_auprc']:.4f}"
)

wandb.log(
    {
        "best/val_macro_auroc": final_metrics["macro_auroc"],
        "best/val_macro_auprc": final_metrics["macro_auprc"],
    }
)

wandb.finish()


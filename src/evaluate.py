import numpy as np
import torch
from tqdm.auto import tqdm
import time
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

from engine import masked_bce_loss


def compute_auroc_auprc(probs, labels, masks, class_names=None):

    num_classes = labels.shape[1]

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    auroc_dict = {}
    auprc_dict = {}

    for c in range(num_classes):

        unmask = masks[:, c] == 1

        if unmask.sum() == 0:
            continue

        true  = labels[unmask, c]
        score = probs[unmask, c]

        if len(np.unique(true)) < 2:
            continue

        auroc_dict[class_names[c]] = roc_auc_score(true, score)
        auprc_dict[class_names[c]] = average_precision_score(true, score)

    macro_auroc = np.mean(list(auroc_dict.values())) if auroc_dict else 0.0
    macro_auprc = np.mean(list(auprc_dict.values())) if auprc_dict else 0.0

    return {
        "macro_auroc": float(macro_auroc),
        "macro_auprc": float(macro_auprc),
        "auroc_per_class": auroc_dict,
        "auprc_per_class": auprc_dict
        }



def find_threshold(probs, targets, masks, target_recall=0.9, step=0.01):
    
    num_classes = targets.shape[1]
    thresholds_dict = {}
    print(f"\n===== Threshold Search (Recall ≥ {target_recall}) =====")
    
    grid = np.arange(0.0, 1.0 + step, step)
    
    for c in range(num_classes):

        unmask = masks[:, c] == 1
        y_true = targets[unmask, c]
        y_score = probs[unmask, c]
        
        best_precision = -1
        best_threshold = None
        
        for thr in grid:
        
            preds = (y_score >= thr).astype(float)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
                
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            
            if recall >= target_recall:
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = thr
    
        if best_threshold is None:
            print(f"Class {c}: No threshold satisfies Recall ≥ {target_recall}")
        else:
            print(
                f"Class {c}: "
                f"Best thr={best_threshold:.3f} | "
                f"Precision={best_precision:.4f}"
            )
            thresholds_dict[c] = best_threshold
    
    return thresholds_dict



def evaluate_model(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
):
    model.eval()

    running_loss_sum = 0.0
    running_unmask = 0.0
    running_correct = 0.0

    all_probs = []
    all_labels = []
    all_masks = []

    start = time.time()

    with torch.no_grad():
        for images, labels, masks in tqdm(loader, desc="Test", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            loss = masked_bce_loss(criterion, logits, labels, masks)

            num_unmask = masks.sum().clamp(min=1.0)
            running_loss_sum += loss.item() * num_unmask.item()
            running_unmask += num_unmask.item()

            preds = (probs >= threshold).float()
            correct = ((preds == labels).float() * masks).sum()
            running_correct += correct.item()

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(masks.cpu())

    avg_loss = running_loss_sum / running_unmask
    acc = running_correct / running_unmask

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    masks_np = torch.cat(all_masks).numpy()

    metrics = compute_auroc_auprc(
        probs=probs_np,
        labels=labels_np,
        masks=masks_np
    )

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_auroc": metrics["macro_auroc"],
        "macro_auprc": metrics["macro_auprc"]
    }



def evaluate_ensemble(
    model1,
    model2,
    loader,
    device,
    alpha=0.5,
    threshold=0.5,
):
    model1.eval()
    model2.eval()

    total_correct = 0.0
    total_count = 0.0

    all_probs = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for images, labels, masks in tqdm(loader, desc="EnsVal", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            masks  = masks.to(device)

            probs1 = torch.sigmoid(model1(images))
            probs2 = torch.sigmoid(model2(images))
            probs  = alpha * probs1 + (1.0 - alpha) * probs2

            preds = (probs >= threshold).float()
            correct = ((preds == labels).float() * masks).sum()
            num_unmask = masks.sum().clamp(min=1.0)

            total_correct += correct.item()
            total_count   += num_unmask.item()

            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(masks.detach().cpu())

    val_acc = total_correct / max(total_count, 1.0)

    probs_np  = torch.cat(all_probs,  dim=0).numpy().astype(np.float32)
    labels_np = torch.cat(all_labels, dim=0).numpy().astype(np.float32)
    masks_np  = torch.cat(all_masks,  dim=0).numpy().astype(np.float32)

    metrics = compute_auroc_auprc(probs=probs_np, labels=labels_np, masks=masks_np)

    return {
        "accuracy": float(val_acc),
        "macro_auroc": float(metrics["macro_auroc"]),
        "macro_auprc": float(metrics["macro_auprc"]),
    }


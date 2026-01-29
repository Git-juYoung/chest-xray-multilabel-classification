import sys
sys.path.insert(0, "src")

import torch
import pandas as pd

from data import prepare_train_val_test, build_train_val_dataloaders, build_test_dataloader
from dataset import CustomDataset
from transforms import build_transforms
from models import build_resnet50, build_efficientnet_b1
from train_utils import get_device
from evaluate import evaluate_ensemble
from config import ensemble_config


def main():
    device = get_device()
    
    root = "images_224"
    
    df = pd.read_csv("label_chexpert.csv")
    
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
    batch_size= 32,
    num_workers= 4,
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
    
    alpha_candidates = ensemble_config["alpha_candidates"]
    val_results = []
    
    for alpha in alpha_candidates:
        results = evaluate_ensemble(
            model1=resnet_model,
            model2=efficient_model,
            loader=val_loader,
            device=device,
            alpha=alpha,
        )
    
        val_results.append({
            "alpha": alpha,
            "accuracy": results["accuracy"],
            "macro_auroc": results["macro_auroc"],
            "macro_auprc": results["macro_auprc"],
        })
    
        print(
            f"[Val] alpha={alpha:.2f} | "
            f"Acc={results['accuracy']:.4f} | "
            f"AUROC={results['macro_auroc']:.4f} | "
            f"AUPRC={results['macro_auprc']:.4f}"
        )
    
    val_results_sorted = sorted(
        val_results,
        key=lambda x: x["macro_auprc"],
        reverse=True
    )
    
    print("\n===== Top 3 (by Val AUPRC) =====")
    for r in val_results_sorted[:3]:
        print(
            f"alpha={r['alpha']:.2f} | "
            f"Acc={r['accuracy']:.4f} | "
            f"AUROC={r['macro_auroc']:.4f} | "
            f"AUPRC={r['macro_auprc']:.4f}"
        )
    
    best_alpha = val_results_sorted[0]["alpha"]
    print(f"\nSelected alpha (best AUPRC): {best_alpha:.2f}")
    
    test_results = evaluate_ensemble(
    model1=resnet_model,
    model2=efficient_model,
    loader=test_loader,
    device=device,
    alpha=best_alpha,
    )
    
    
    print("\n===== Test Results (Ensemble) =====")
    print(f"Alpha : {best_alpha:.2f}")
    print(f"Accuracy : {test_results['accuracy']:.4f}")
    print(f"Macro AUROC : {test_results['macro_auroc']:.4f}")
    print(f"Macro AUPRC : {test_results['macro_auprc']:.4f}")
    
if __name__ == "__main__":
    main()
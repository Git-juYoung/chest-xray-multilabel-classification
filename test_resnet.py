import sys
sys.path.insert(0, "src")

import pandas as pd
import torch

from seed import set_seed
from config import r_config
from data import prepare_train_val_test, build_test_dataloader
from dataset import CustomDataset
from transforms import build_transforms
from models import build_resnet50
from train_utils import get_device, build_criterion
from evaluate import evaluate_model


def main():
    set_seed()
    
    CSV_PATH = "label_chexpert.csv"
    IMAGE_ROOT = "images_224"
    
    df = pd.read_csv(CSV_PATH)
    
    (
        _, _, test_df,
        _, _, test_path,
        _, _, test_target,
        _, _, test_mask,
        _
    ) = prepare_train_val_test(
        df=df,
        root=IMAGE_ROOT,
        random_state=42,
        verbose=True
    )
    
    _, val_test_transform = build_transforms()
    
    test_dataset = CustomDataset(
        path=test_path,
        target=test_target,
        mask=test_mask,
        transform=val_test_transform,
    )
    
    test_loader = build_test_dataloader(
        test_dataset=test_dataset,
        batch_size=r_config["batch_size"],
        num_workers=r_config["num_workers"],
        pin_memory=True,
    )
    
    
    device = get_device()
    
    num_classes = test_target.shape[1]
    
    model = build_resnet50(
        unfreeze_from=r_config["unfreeze_from"],
        num_classes=num_classes,
    ).to(device)
    
    ckpt_path = "best_resnet.pth"
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
    
    criterion = build_criterion()
    
    results = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    print("===== Test Results (ResNet) =====")
    print(f"Loss        : {results['loss']:.4f}")
    print(f"Accuracy    : {results['accuracy']:.4f}")
    print(f"Macro AUROC : {results['macro_auroc']:.4f}")
    print(f"Macro AUPRC : {results['macro_auprc']:.4f}")
    
if __name__ == "__main__":
    main()
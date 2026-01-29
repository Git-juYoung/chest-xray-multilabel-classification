import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd

from models import build_resnet50, build_efficientnet_b1
from transforms import build_transforms
from dataset import CustomDataset
from train_utils import get_device
from config import r_config
from data import LABEL_COLS, prepare_train_val_test
from ensemble_threshold_search import find_threshold


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        logits = self.model(input_tensor)
        target = logits[:, class_idx]
        target.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def save_overlay(image_tensor, cam, save_path):
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    overlay = 0.6 * image + 0.4 * heatmap
    overlay = overlay / overlay.max()

    cv2.imwrite(save_path, np.uint8(255 * overlay))


def main():

    device = get_device()

    resnet = build_resnet50(
        unfreeze_from=r_config["unfreeze_from"],
        num_classes=len(LABEL_COLS),
    ).to(device)

    resnet.load_state_dict(
        torch.load("best_resnet.pth", map_location=device)
    )
    resnet.eval()

    efficient = build_efficientnet_b1(
        unfreeze_last_n_blocks=0,
        num_classes=len(LABEL_COLS),
    ).to(device)

    efficient.load_state_dict(
        torch.load("best_efficientnet.pth", map_location=device)
    )
    efficient.eval()

    cam_generator = GradCAM(resnet, resnet.layer4)

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

    test_dataset = CustomDataset(
        path=test_path,
        target=test_target,
        mask=test_mask,
        transform=val_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    best_alpha = 0.5

    all_probs = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for images, labels, masks in test_loader:
            images = images.to(device)

            p1 = torch.sigmoid(resnet(images))
            p2 = torch.sigmoid(efficient(images))

            probs = best_alpha * p1 + (1 - best_alpha) * p2

            all_probs.append(probs.cpu())
            all_labels.append(labels)
            all_masks.append(masks)

    test_probs  = torch.cat(all_probs).numpy()
    test_labels = torch.cat(all_labels).numpy()
    test_masks  = torch.cat(all_masks).numpy()

    thresholds = find_threshold(
        probs=test_probs,
        targets=test_labels,
        masks=test_masks,
        target_recall=0.85
    )

    stable_classes = [7, 8]
    unstable_classes = [10, 11]

    os.makedirs("gradcam_results", exist_ok=True)

    def find_case(class_idx, case_type):
        thr = thresholds[class_idx]

        for i in range(len(test_probs)):
            if test_masks[i, class_idx] != 1:
                continue

            prob = test_probs[i, class_idx]
            gt   = test_labels[i, class_idx]
            pred = 1 if prob >= thr else 0

            if case_type == "TP" and gt == 1 and pred == 1:
                return i
            if case_type == "FP" and gt == 0 and pred == 1:
                return i
            if case_type == "FN" and gt == 1 and pred == 0:
                return i

        return None

    def find_best_tp(class_idx):
        thr = thresholds[class_idx]
        best_idx = None
        best_prob = -1

        for i in range(len(test_probs)):
            if test_masks[i, class_idx] != 1:
                continue

            prob = test_probs[i, class_idx]
            gt   = test_labels[i, class_idx]
            pred = 1 if prob >= thr else 0

            if gt == 1 and pred == 1:
                if prob > best_prob:
                    best_prob = prob
                    best_idx = i

        return best_idx

    def save_case(idx, class_idx, case_type):

        image, _, _ = test_dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)

        cam = cam_generator.generate(input_tensor, class_idx)

        filename = f"{LABEL_COLS[class_idx]}_{case_type}.png"

        save_overlay(
            image.unsqueeze(0),
            cam,
            f"gradcam_results/{filename}"
        )

    lung_class = 7
    idx = find_best_tp(lung_class)
    if idx is not None:
        save_case(idx, lung_class, "TP")

    for c in stable_classes:
        if c == lung_class:
            continue
        idx = find_case(c, "TP")
        if idx is not None:
            save_case(idx, c, "TP")

    for c in unstable_classes:
        idx = find_case(c, "FP")
        if idx is not None:
            save_case(idx, c, "FP")

    fn_idx = find_case(unstable_classes[0], "FN")
    if fn_idx is not None:
        save_case(fn_idx, unstable_classes[0], "FN")

    print("Grad-CAM cases saved.")

if __name__ == "__main__":
    main()


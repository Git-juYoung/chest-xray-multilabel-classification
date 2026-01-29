import sys
sys.path.insert(0, "src")

import pandas as pd
import numpy as np

from data import prepare_train_val_test, LABEL_COLS


df = pd.read_csv("label_chexpert.csv")

(
    train, val, test,
    train_path, val_path, test_path,
    train_target, val_target, test_target,
    train_mask, val_mask, test_mask,
    pos_weight
) = prepare_train_val_test(df, root="YOUR_IMAGE_ROOT", verbose=True)


print("\n===== Split Ratio =====")
total = len(train) + len(val) + len(test)
print("train:", len(train) / total)
print("val  :", len(val) / total)
print("test :", len(test) / total)


def print_stats(name, target, mask):
    P = (target * mask).sum(axis=0)
    N = mask.sum(axis=0) - P
    valid = mask.sum(axis=0)

    print(f"\n===== {name} =====")
    print("P per class:\n", dict(zip(LABEL_COLS, P.astype(int))))
    print("P+N per class:\n", dict(zip(LABEL_COLS, valid.astype(int))))


print_stats("Train", train_target, train_mask)
print_stats("Val", val_target, val_mask)
print_stats("Test", test_target, test_mask)


train_P = (train_target * train_mask).sum(axis=0)
print("\nMinimum P in train:", int(train_P.min()))

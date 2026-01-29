import os
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader



LABEL_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def prepare_train_val_test(df, root, random_state=42, verbose=True):
    subjects = df["subject_id"].unique()

    train_subj, temp_subj = train_test_split(
        subjects,
        test_size=0.2,
        random_state=random_state,
    )

    val_subj, test_subj = train_test_split(
        temp_subj,
        test_size=0.5,
        random_state=random_state,
    )

    train = df[df["subject_id"].isin(train_subj)].reset_index(drop=True)
    val = df[df["subject_id"].isin(val_subj)].reset_index(drop=True)
    test = df[df["subject_id"].isin(test_subj)].reset_index(drop=True)

    if verbose:
        print("train ratio:", len(train) / len(df))
        print("val ratio  :", len(val) / len(df))
        print("test ratio :", len(test) / len(df))

    train_labels = train[LABEL_COLS]
    val_labels = val[LABEL_COLS]
    test_labels = test[LABEL_COLS]

    train_mask = (train_labels.notna() & (train_labels != -1.0)).astype(float).values
    val_mask = (val_labels.notna() & (val_labels != -1.0)).astype(float).values
    test_mask = (test_labels.notna() & (test_labels != -1.0)).astype(float).values

    train_target = (train_labels == 1.0).astype(float).values
    val_target = (val_labels == 1.0).astype(float).values
    test_target = (test_labels == 1.0).astype(float).values

    P = (train_target * train_mask).sum(axis=0)
    N = train_mask.sum(axis=0) - P

    pos_weight = N / P

    train_path = [
        os.path.join(
            root,
            f"p{str(row.subject_id)[:2]}/p{row.subject_id}/s{row.study_id}/{row.dicom_id}.jpg",
        )
        for row in train.itertuples()
    ]
    val_path = [
        os.path.join(
            root,
            f"p{str(row.subject_id)[:2]}/p{row.subject_id}/s{row.study_id}/{row.dicom_id}.jpg",
        )
        for row in val.itertuples()
    ]
    test_path = [
        os.path.join(
            root,
            f"p{str(row.subject_id)[:2]}/p{row.subject_id}/s{row.study_id}/{row.dicom_id}.jpg",
        )
        for row in test.itertuples()
    ]

    return (
        train, val, test,
        train_path, val_path, test_path,
        train_target, val_target, test_target,
        train_mask, val_mask, test_mask,
        pos_weight
    )

def build_train_val_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    pin_memory,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_test_dataloader(
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader


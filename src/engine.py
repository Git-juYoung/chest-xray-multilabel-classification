import torch
from tqdm.auto import tqdm


def masked_bce_loss(criterion, logits, labels, masks):
    loss = criterion(logits, labels)
    loss = loss * masks
    loss = loss.sum()
    num_unmask = masks.sum().clamp(min=1.0)
    loss = loss / num_unmask
    return loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    name,
    epoch,
    num_epochs,
):
    model.train()
    train_loss = 0.0
    train_batches = 0

    train_bar = tqdm(
        loader,
        desc=f"{name} Epoch {epoch}/{num_epochs} [Train]",
        leave=False
    )

    for images, labels, masks in train_bar:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = masked_bce_loss(criterion, outputs, labels, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

    avg_train_loss = train_loss / train_batches
    return avg_train_loss


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    name,
    epoch,
    num_epochs,
):
    model.eval()
    val_loss = 0.0
    val_batches = 0

    val_bar = tqdm(
        loader,
        desc=f"{name} Epoch {epoch}/{num_epochs} [Val]",
        leave=False
    )

    with torch.no_grad():
        for images, labels, masks in val_bar:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = masked_bce_loss(criterion, outputs, labels, masks)

            val_loss += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches
    return avg_val_loss


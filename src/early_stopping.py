import torch

class EarlyStopping:
    def __init__(self, patience, save_path):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            print(f"Best model saved! val_loss={val_loss:.4f}")
            return False

        self.counter += 1
        print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            print("Early stopping triggered.")
            return True
        return False


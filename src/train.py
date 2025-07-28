import torch
from tqdm import tqdm
import copy

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20, patience=3, save_path="outputs/best_model.pth"):
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_counter = 0
            print("✅ New best model saved.")
            torch.save(best_model_wts, save_path)
        else:
            no_improve_counter += 1
            print(f"⚠️ No improvement. Early stop counter: {no_improve_counter}/{patience}")
            if no_improve_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return train_losses, val_losses

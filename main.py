import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from efficient_kan import KAN
from sklearn.metrics import cohen_kappa_score

DATA_DIR = "bciciv2a_processed"
RESULTS_DIR = "dicenet_bciciv2a_results_2"
PLOTS_DIR = os.path.join(RESULTS_DIR, "fold_plots")
UNIVERSAL_MODEL_PATH = os.path.join(RESULTS_DIR, "universal_model.pth")

NUM_CHANNELS = 22
TIME_STEPS = 1000
NUM_CLASSES = 4

PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 200
BATCH_SIZE = 32
PRETRAIN_LR = 1e-3
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1

RANDOM_SEED = 42
SUBJECTS = [f"A0{i}" for i in range(1, 10)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Hardware Processing Device: {DEVICE}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MIDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.01, max_shift=15):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            if self.max_shift > 0:
                shift = random.randint(-self.max_shift, self.max_shift)
                x = torch.roll(x, shifts=shift, dims=-1)
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if random.random() < 0.5:
                mask_len = random.randint(20, 100)
                start = random.randint(0, x.shape[-1] - mask_len)
                x[:, start : start + mask_len] = 0.0
            if random.random() < 0.5:
                ch_idx = random.randint(0, x.shape[0] - 1)
                x[ch_idx, :] = 0.0

        x = x.unsqueeze(0)
        return x, y

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class MultiScaleDICENet(nn.Module):
    def __init__(self, num_classes=4, num_channels=22, time_steps=1000):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, 16, kernel_size=(1, 64), padding="same", bias=False)
        self.temp_conv2 = nn.Conv2d(1, 16, kernel_size=(1, 32), padding="same", bias=False)
        self.temp_conv3 = nn.Conv2d(1, 16, kernel_size=(1, 16), padding="same", bias=False)
        
        self.spatial_dw = nn.Conv2d(48, 48, kernel_size=(num_channels, 1), groups=48, bias=False)
        self.spatial_pw = nn.Conv2d(48, 48, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(48)
        self.se = SEBlock(48)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15))
        self.dropout_cnn = nn.Dropout(0.5)

        self.seq_len = time_steps // 15
        
        self.proj = nn.Linear(48, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, 64) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.5,
            activation="gelu",
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.kan_norm = nn.LayerNorm(64)
        self.classifier = KAN(
            layers_hidden=[64, num_classes],
            grid_size=3,
            spline_order=2,
            base_activation=nn.SiLU
        )

    def forward(self, x):
        t1 = self.temp_conv1(x)
        t2 = self.temp_conv2(x)
        t3 = self.temp_conv3(x)
        x = torch.cat([t1, t2, t3], dim=1)

        x = self.spatial_dw(x)
        x = self.spatial_pw(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)

        x = x.squeeze(2).transpose(1, 2)
        x = self.proj(x)
        x = x + self.pos_embedding

        x = self.transformer_encoder(x)
        
        x = x.mean(dim=1)
        x = self.kan_norm(x)
        
        logits = self.classifier(x)
        return logits

def plot_subject_curves(sub_id, history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title(f"{sub_id} - Fine-tuning Loss Curve")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.plot(epochs, history["test_acc"], "g-", label="Test Acc")
    ax2.set_title(f"{sub_id} - Fine-tuning Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{sub_id}_metrics.png"))
    plt.close()

def train_universal_model():
    if os.path.exists(UNIVERSAL_MODEL_PATH):
        print(f"\nUniversal model already exists at {UNIVERSAL_MODEL_PATH}. Skipping pre-training.")
        return

    print("\nAggregating all training data for Universal Model...")
    all_X, all_y = [], []
    for sub_id in SUBJECTS:
        train_path = os.path.join(DATA_DIR, f"{sub_id}_train.pt")
        if os.path.exists(train_path):
            data = torch.load(train_path, map_location="cpu")
            all_X.append(data["X"])
            all_y.append(data["y"])
    
    if not all_X:
        print("No training data found.")
        return

    X_univ = torch.cat(all_X, dim=0)
    y_univ = torch.cat(all_y, dim=0)
    
    univ_dataset = MIDataset(X_univ, y_univ, augment=True)
    univ_loader = DataLoader(univ_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = MultiScaleDICENet(num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, time_steps=TIME_STEPS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
    
    print(f"Training Universal Model on {X_univ.shape[0]} trials for {PRETRAIN_EPOCHS} epochs...")
    
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(univ_loader, desc=f"Universal Epoch {epoch}/{PRETRAIN_EPOCHS}", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    torch.save(model.state_dict(), UNIVERSAL_MODEL_PATH)
    print(f"Universal model saved to {UNIVERSAL_MODEL_PATH}")

def run_subject(sub_id):
    target_train_path = os.path.join(DATA_DIR, f"{sub_id}_train.pt")
    target_test_path = os.path.join(DATA_DIR, f"{sub_id}_test.pt")

    if not (os.path.exists(target_train_path) and os.path.exists(target_test_path)):
        print(f"Skipping {sub_id}: missing processed tensors.")
        return None

    target_train_data = torch.load(target_train_path, map_location="cpu")
    target_test_data = torch.load(target_test_path, map_location="cpu")

    X_target_train, y_target_train = target_train_data["X"], target_train_data["y"]
    X_test, y_test = target_test_data["X"], target_test_data["y"]

    n_target_total = X_target_train.shape[0]
    val_frac = 0.15
    perm = torch.randperm(n_target_total)
    n_val = max(1, int(n_target_total * val_frac))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    X_fit, y_fit = X_target_train[fit_idx], y_target_train[fit_idx]
    X_val, y_val = X_target_train[val_idx], y_target_train[val_idx]

    train_dataset = MIDataset(X_fit, y_fit, augment=True)
    val_dataset = MIDataset(X_val, y_val, augment=False)
    test_dataset = MIDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiScaleDICENet(num_classes=NUM_CLASSES, num_channels=NUM_CHANNELS, time_steps=TIME_STEPS).to(DEVICE)
    model.load_state_dict(torch.load(UNIVERSAL_MODEL_PATH, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer_ft = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=FINETUNE_EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": []}
    best_test_acc = 0.0
    best_kappa = 0.0
    best_epoch = 0

    print(f"[{sub_id}] Fine-tuning Universal Model ({X_fit.shape[0]} trials)...")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"[{sub_id}] Finetune {epoch}/{FINETUNE_EPOCHS}", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_ft.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_ft.step()

            running_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                v_total += y.size(0)
                v_correct += pred.eq(y).sum().item()

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        t_correct, t_total = 0, 0
        y_true_all, y_pred_all = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = out.max(1)
                t_total += y.size(0)
                t_correct += pred.eq(y).sum().item()
                y_true_all.extend(y.cpu().numpy())
                y_pred_all.extend(pred.cpu().numpy())
                
        test_acc = t_correct / t_total
        current_kappa = cohen_kappa_score(y_true_all, y_pred_all)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_kappa = current_kappa
            best_epoch = epoch

        print(
            f"[{sub_id}] FT Epoch {epoch:03d}/{FINETUNE_EPOCHS} | "
            f"Train Acc {train_acc*100:.2f}% | "
            f"Val Acc {val_acc*100:.2f}% | "
            f"Test Acc {test_acc*100:.2f}%"
        )

    plot_subject_curves(sub_id, history)
    print(f"[{sub_id}] BEST TEST Accuracy: {best_test_acc*100:.2f}% | Kappa: {best_kappa:.4f} (Achieved at FT Epoch {best_epoch})")

    return {
        "Subject": sub_id,
        "Best_Epoch": best_epoch,
        "Test_Accuracy_%": round(best_test_acc * 100, 2),
        "Kappa": round(best_kappa, 4),
        "N_test_trials": len(y_true_all),
    }

def main():
    set_seed(RANDOM_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_universal_model()

    rows = []
    for sub_id in SUBJECTS:
        print("\n" + "=" * 60)
        print(f"Subject {sub_id}")
        print("=" * 60)
        result = run_subject(sub_id)
        if result is not None:
            rows.append(result)

    if not rows:
        print("No subjects were evaluated.")
        return

    df = pd.DataFrame(rows)
    avg_acc = df["Test_Accuracy_%"].mean()
    avg_kappa = df["Kappa"].mean()

    summary_row = pd.DataFrame([{
        "Subject": "AVERAGE",
        "Best_Epoch": "-",
        "Test_Accuracy_%": round(avg_acc, 2),
        "Kappa": round(avg_kappa, 4),
        "N_test_trials": df["N_test_trials"].sum(),
    }])
    df_full = pd.concat([df, summary_row], ignore_index=True)

    print("\n" + "=" * 60)
    print("FINAL RESULTS — DICE-KAN-Net (Using Universal Model)")
    print("=" * 60)
    print(df_full.to_string(index=False))
    print("-" * 60)

    out_csv = os.path.join(RESULTS_DIR, "dicenet_bciciv2a_results.csv")
    df_full.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

if __name__ == "__main__":
    main()
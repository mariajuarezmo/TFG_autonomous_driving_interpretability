"""
PilotNet (Mejorado) con modificación de la función de pérdida:
- SmoothL1 / Huber (robusta a outliers)
- Pérdida ponderada para dar más peso a frames de esquiva

Objetivo: mejorar rendimiento en maniobras de esquiva sin modificar el dataset,
solo el criterio de optimización.

Salida:
- split_indices.pt (si no existe, se crea; si existe, se reutiliza)
- pilotnet_loss_modificada.pth (mejor checkpoint por MAE en esquivas)
"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



# CONFIGURACIÓN

torch.manual_seed(626)
torch.cuda.manual_seed_all(626)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

processed_data_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_processed"
DATA_FILE = "processed_data_v3.pt"

num_epochs = 30
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-5

# Definición de esquiva
# Definición de esquiva (TOP 10% por percentil)
ESQUIVA_PERCENTIL = 0.90       # top 10%
ESQUIVA_THRESHOLD = None
ESQUIVA_LOSS_WEIGHT = 3.0     # peso adicional en la pérdida para muestras esquiva

# Error relativo robusto
TORQUE_EPS = 5.0              # evita explosión del error relativo cuando real ~ 0

# Split fijo para comparativas
SPLIT_FILE = "split_indices.pt"

# Salida
CKPT_OUT = "pilotnet_loss_modificada.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PILOTNET - LOSS MODIFICADA (SmoothL1 ponderada en esquivas)")
print("=" * 70)
print(f"Dispositivo: {device}")
print(f"Archivo de datos: {DATA_FILE}")
print(f"ESQUIVA_PERCENTIL: {int(ESQUIVA_PERCENTIL*100)}")
print(f"ESQUIVA_LOSS_WEIGHT: {ESQUIVA_LOSS_WEIGHT}")
print("=" * 70)



# DESNORMALIZACIÓN

def create_denormalize_function(normalization_params):
    """Crea función de desnormalización según método usado en el .pt."""
    if normalization_params is None:
        # fallback: normalización por max_abs
        def denorm(norm_val, max_val):
            return norm_val * max_val
        return denorm

    method = normalization_params.get("method", "unknown")
    if method == "minmax_symmetric":
        min_torque = normalization_params["min_torque"]
        range_torque = normalization_params["range"]

        def denorm(norm_val, _=None):
            return (norm_val + 1) * range_torque / 2 + min_torque

        return denorm

    # fallback
    def denorm(norm_val, max_val):
        return norm_val * max_val
    return denorm



# CARGA DE DATOS

data_path = os.path.join(processed_data_dir, DATA_FILE)
if not os.path.exists(data_path):
    raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

data = torch.load(data_path)
images_tensor = data["images"]
torques_tensor = data["torques"]
num_samples = data["num_samples"]

normalization_params = data.get("normalization_params", None)
denormalize = create_denormalize_function(normalization_params)

if "original_stats" in data:
    max_torque_ref = data["original_stats"]["max"]
    min_torque_ref = data["original_stats"]["min"]
else:
    max_torque_ref = data["max_torque"]
    min_torque_ref = data.get("min_torque", -max_torque_ref)

print(f"\nDatos cargados: {num_samples} muestras")
print(f"Imágenes: {tuple(images_tensor.shape)}")
print(f" Torques (norm): {tuple(torques_tensor.shape)}")
print(f" Torque real rango: [{min_torque_ref:.2f}, {max_torque_ref:.2f}]")


# DEFINICIÓN DE ESQUIVA POR PERCENTIL (TOP 10%)

torques_real_all = denormalize(torques_tensor, max_torque_ref)
abs_torque = torch.abs(torques_real_all)

ESQUIVA_THRESHOLD = torch.quantile(abs_torque, ESQUIVA_PERCENTIL).item()

print(f"\nDefinición de esquiva por percentil:")
print(f"   Percentil: {int(ESQUIVA_PERCENTIL * 100)}")
print(f"   Umbral |torque| >= {ESQUIVA_THRESHOLD:.2f}")

# Comprobación de porcentaje real
esquiva_mask_all = abs_torque >= ESQUIVA_THRESHOLD
print(f"   % esquivas reales (dataset completo): {esquiva_mask_all.float().mean().item() * 100:.2f}%\n")


# SPLIT FIJO PARA COMPARATIVAS

if os.path.exists(SPLIT_FILE):
    split = torch.load(SPLIT_FILE)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    print(f" Usando split fijo existente: {SPLIT_FILE}")
else:
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    torch.save({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_FILE)
    print(f" Split creado y guardado en: {SPLIT_FILE}")

train_images = images_tensor[train_idx].to(device)
train_torques = torques_tensor[train_idx].to(device)
test_images = images_tensor[test_idx].to(device)
test_torques = torques_tensor[test_idx].to(device)

print(f"   Train: {len(train_idx)} ({len(train_idx)/num_samples*100:.1f}%)")
print(f"   Test:  {len(test_idx)} ({len(test_idx)/num_samples*100:.1f}%)\n")



# MODELO

class PilotNetMejorado(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(36)

        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 1 * 18, 100)
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(100, 50)
        self.bn_fc2 = nn.BatchNorm1d(50)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(50, 10)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, start_dim=1)

        x = self.dropout1(torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn_fc2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.fc3(x)))

        return self.fc4(x)  # salida lineal


model = PilotNetMejorado().to(device)
print(f" Modelo creado. Parámetros: {sum(p.numel() for p in model.parameters()):,}\n")



# LOSS: SmoothL1 (Huber) + ponderación por esquiva

base_loss = nn.SmoothL1Loss(reduction="none")

def weighted_huber_loss(pred_norm_2d: torch.Tensor, target_norm_2d: torch.Tensor) -> torch.Tensor:
    """
    pred_norm_2d, target_norm_2d: shape [B,1] (normalizados)
    etiqueta esquiva según torque real del target y aplica mayor peso.
    """
    # torque real SOLO para definir máscara
    real_t = denormalize(target_norm_2d.squeeze(1).detach().cpu(), max_torque_ref)
    esquiva_mask = (torch.abs(real_t) >= ESQUIVA_THRESHOLD).to(pred_norm_2d.device)

    weights = torch.ones_like(target_norm_2d.squeeze(1), device=pred_norm_2d.device)
    weights[esquiva_mask] = ESQUIVA_LOSS_WEIGHT

    per_sample = base_loss(pred_norm_2d.squeeze(1), target_norm_2d.squeeze(1))
    return (weights * per_sample).mean()



# MÉTRICAS

@torch.no_grad()
def calcular_metricas(preds_norm: torch.Tensor, reales_norm: torch.Tensor) -> dict:
    """
    preds_norm y reales_norm: tensores 1D en escala normalizada.
    Devuelve métricas globales y específicas de esquiva.
    """
    preds_t = denormalize(preds_norm.cpu(), max_torque_ref)
    reales_t = denormalize(reales_norm.cpu(), max_torque_ref)

    mse_norm = torch.mean((preds_norm.cpu() - reales_norm.cpu()) ** 2).item()

    mse_t = torch.mean((preds_t - reales_t) ** 2).item()
    rmse_t = float(np.sqrt(mse_t))
    mae_t = torch.mean(torch.abs(preds_t - reales_t)).item()

    den = torch.clamp(torch.abs(reales_t), min=TORQUE_EPS)
    er = (torch.abs(preds_t - reales_t) / den) * 100.0
    er_mean = er.mean().item()
    er_median = er.median().item()

    sign_ok = ((preds_t > 0) == (reales_t > 0)).float().mean().item() * 100.0

    esquiva_mask = torch.abs(reales_t) >= ESQUIVA_THRESHOLD
    if esquiva_mask.any():
        mae_esq = torch.mean(torch.abs(preds_t[esquiva_mask] - reales_t[esquiva_mask])).item()
        sign_ok_esq = ((preds_t[esquiva_mask] > 0) == (reales_t[esquiva_mask] > 0)).float().mean().item() * 100.0
        frac_esq = esquiva_mask.float().mean().item() * 100.0
    else:
        mae_esq = float("nan")
        sign_ok_esq = float("nan")
        frac_esq = 0.0

    return {
        "mse_norm": mse_norm,
        "rmse_torque": rmse_t,
        "mae_torque": mae_t,
        "er_mean": er_mean,
        "er_median": er_median,
        "sign_ok": sign_ok,
        "mae_esquiva": mae_esq,
        "sign_ok_esquiva": sign_ok_esq,
        "frac_esquiva_test": frac_esq
    }



# DATALOADERS

train_loader = DataLoader(
    TensorDataset(train_images, train_torques),
    batch_size=batch_size,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(test_images, test_torques),
    batch_size=batch_size,
    shuffle=False
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)



# ENTRENAMIENTO

train_loss_hist = []
test_mse_hist = []
mae_hist = []
mae_esq_hist = []
er_med_hist = []

best_test_mae_esq = float("inf")
best_epoch = 0
best_metrics = None

print("-" * 70)
print("Entrenando...")
print("-" * 70)

for epoch in range(num_epochs):
    # ---- TRAIN
    model.train()
    train_loss = 0.0

    for imgs, torq in train_loader:
        imgs = imgs.to(device)
        torq = torq.to(device).unsqueeze(1)

        optimizer.zero_grad()
        out = model(imgs)
        loss = weighted_huber_loss(out, torq)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- TEST (recolectar preds)
    model.eval()
    test_mse_norm = 0.0
    preds_all, reals_all = [], []

    with torch.no_grad():
        for imgs, torq in test_loader:
            imgs = imgs.to(device)
            torq = torq.to(device).unsqueeze(1)
            out = model(imgs)

            test_mse_norm += torch.mean((out - torq) ** 2).item() * imgs.size(0)
            preds_all.append(out.squeeze(1).cpu())
            reals_all.append(torq.squeeze(1).cpu())

    test_mse_norm /= len(test_loader.dataset)
    scheduler.step(test_mse_norm)

    preds = torch.cat(preds_all)
    reals = torch.cat(reals_all)
    m = calcular_metricas(preds, reals)

    # ---- Guardar historia
    train_loss_hist.append(train_loss)
    test_mse_hist.append(test_mse_norm)
    mae_hist.append(m["mae_torque"])
    mae_esq_hist.append(m["mae_esquiva"])
    er_med_hist.append(m["er_median"])

    # ---- Mejor checkpoint por MAE en esquivas
    if not np.isnan(m["mae_esquiva"]) and m["mae_esquiva"] < best_test_mae_esq:
        best_test_mae_esq = m["mae_esquiva"]
        best_epoch = epoch + 1
        best_metrics = m

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "normalization_params": normalization_params,
            "max_torque_ref": max_torque_ref,
            "min_torque_ref": min_torque_ref,
            "data_file": DATA_FILE,
            "split_file": SPLIT_FILE,
            "loss": "SmoothL1(weighted)",
            "esquiva_threshold": ESQUIVA_THRESHOLD,
            "esquiva_percentil": ESQUIVA_PERCENTIL,
            "esquiva_loss_weight": ESQUIVA_LOSS_WEIGHT,
            "best_mae_esquiva": best_test_mae_esq,
            "best_metrics": best_metrics
        }, CKPT_OUT)

    
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"TrainLoss: {train_loss:.6f} | "
            f"TestMSE(norm): {test_mse_norm:.6f} | "
            f"MAE: {m['mae_torque']:.2f} | "
            f"MAE_esq: {m['mae_esquiva']:.2f} | "
            f"ER_med: {m['er_median']:.1f}% | "
            f"SignOK: {m['sign_ok']:.1f}% | "
            f"SignOK_esq: {m['sign_ok_esquiva']:.1f}%"
        )

print("-" * 70)
print(f" Mejor modelo (por MAE esquivas) en epoch {best_epoch} | MAE_esq: {best_test_mae_esq:.2f}")
print(f"  Checkpoint: {CKPT_OUT}")
print("-" * 70)

if best_metrics is not None:
    print("\nMétricas del mejor checkpoint:")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")



# PLOT RESUMEN 

fig = plt.figure(figsize=(14, 8))

ax1 = plt.subplot(2, 2, 1)
plt.plot(train_loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Train Loss (SmoothL1 weighted)")
plt.title("Train Loss")
plt.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 2, 2)
plt.plot(test_mse_hist)
plt.xlabel("Epoch")
plt.ylabel("Test MSE (norm)")
plt.title("Test MSE (norm)")
plt.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 2, 3)
plt.plot(mae_hist, label="MAE global")
plt.plot(mae_esq_hist, label="MAE esquivas")
plt.xlabel("Epoch")
plt.ylabel("MAE (torque)")
plt.title("MAE global vs esquivas")
plt.legend()
plt.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 2, 4)
plt.plot(er_med_hist)
plt.xlabel("Epoch")
plt.ylabel("Error relativo (mediana) %")
plt.title("Error relativo (mediana)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pilotnet_loss_modificada.png", dpi=150)
print("\n Figura guardada: pilotnet_loss_modificada.png")

"""
PilotNet (Mejorado) con modificación de la función de pérdida:
- SmoothL1 / Huber (robusta a outliers)
- Pérdida ponderada para dar más peso a frames de esquiva

Preprocesamiento integrado: lee directamente desde Dataset/runN/
sin necesidad de generar processed_data.pt intermedio.

Salida:
- split_indices.pt (si no existe, se crea; si existe, se reutiliza)
- pilotnet_loss_modificada.pth (mejor checkpoint por MAE en esquivas)
"""

import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# ========== CONFIGURACIÓN ==========

torch.manual_seed(626)
torch.cuda.manual_seed_all(626)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Carpeta raíz que contiene las subcarpetas run1, run2, run3, run4
base_data_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\Dataset"
output_dir    = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_processed"

run_folders = ["run1", "run2", "run3", "run4"]

num_epochs    = 40
batch_size    = 64
learning_rate = 1e-3
weight_decay  = 1e-5

# Definición de esquiva (TOP 10% por percentil)
ESQUIVA_PERCENTIL   = 0.90
ESQUIVA_THRESHOLD   = None   # se calcula tras cargar los torques
ESQUIVA_LOSS_WEIGHT = 5.0    # peso adicional en la pérdida para muestras esquiva

# Error relativo robusto
TORQUE_EPS = 5.0

# Split fijo para comparativas
SPLIT_FILE = "split_indices.pt"

# Salida
CKPT_OUT = "pilotnet_loss_modificada.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("PILOTNET - LOSS MODIFICADA (SmoothL1 ponderada en esquivas)")
print("Estructura: Dataset/runN/telemetry_data/ + video_data/")
print("=" * 70)
print(f"Dispositivo:        {device}")
print(f"ESQUIVA_PERCENTIL:  {int(ESQUIVA_PERCENTIL*100)}")
print(f"ESQUIVA_LOSS_WEIGHT:{ESQUIVA_LOSS_WEIGHT}")
print("=" * 70)


# ========== CARGA DE RUTAS Y TORQUES ==========
all_images  = []
all_torques = []

print("\n[1/5] Cargando rutas de imágenes desde carpetas...")

for run_name in run_folders:
    run_path = os.path.join(base_data_dir, run_name)

    if not os.path.isdir(run_path):
        print(f"   ⚠ Carpeta no encontrada, omitiendo: {run_path}")
        continue

    telemetry_dir    = os.path.join(run_path, "telemetry_data")
    video_dir        = os.path.join(run_path, "video_data")
    frame_torque_txt = os.path.join(telemetry_dir, "frame-torque.txt")

    if not os.path.exists(frame_torque_txt):
        print(f"   ⚠ No se encontró frame-torque.txt en {telemetry_dir}, omitiendo {run_name}")
        continue

    if not os.path.isdir(video_dir):
        print(f"   ⚠ No se encontró video_data en {run_path}, omitiendo {run_name}")
        continue

    count_before = len(all_images)

    with open(frame_torque_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_name, torque = parts
            img_path = os.path.join(video_dir, img_name)
            if os.path.exists(img_path):
                all_images.append(img_path)
                all_torques.append(float(torque))

    print(f"   ✓ {run_name}: {len(all_images) - count_before} imágenes cargadas")

print(f"\n   ✓ Total de imágenes encontradas: {len(all_images)}")

if len(all_images) == 0:
    raise RuntimeError("No se encontraron imágenes. Revisa las rutas y la estructura de carpetas.")


# ========== ANÁLISIS Y NORMALIZACIÓN DE TORQUES ==========
print("\n[2/5] Analizando y normalizando torques...")

torques_tensor = torch.tensor(all_torques, dtype=torch.float32)

min_torque  = torques_tensor.min().item()
max_torque  = torques_tensor.max().item()
mean_torque = torques_tensor.mean().item()
std_torque  = torques_tensor.std().item()

# Estas variables sustituyen a max_torque_ref y min_torque_ref del original
max_torque_ref = max_torque
min_torque_ref = min_torque

print(f"   Min:  {min_torque:8.2f}")
print(f"   Max:  {max_torque:8.2f}")
print(f"   Mean: {mean_torque:8.2f}")
print(f"   Std:  {std_torque:8.2f}")

# Normalización Min-Max simétrica [-1, 1]
range_torque       = max_torque - min_torque
torques_normalized = 2 * (torques_tensor - min_torque) / range_torque - 1

normalization_params = {
    'method':     'minmax_symmetric',
    'min_torque': min_torque,
    'max_torque': max_torque,
    'range':      range_torque
}

print(f"\n   Rango normalizado: [{torques_normalized.min():.3f}, {torques_normalized.max():.3f}]")
print(f"   Balance: {abs(torques_normalized.max() + torques_normalized.min()):.6f} (debe ser ≈0)")

# Guardar configuración ligera en .txt
config_file = os.path.join(output_dir, "preprocessing_config_loss_modificada.txt")
with open(config_file, "w") as f:
    f.write("CONFIGURACIÓN DEL PREPROCESAMIENTO (LOSS MODIFICADA - TODOS LOS RUNS)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Directorio origen: {base_data_dir}\n")
    f.write(f"Runs incluidos: {', '.join(run_folders)}\n")
    f.write(f"Total de imágenes: {len(all_images)}\n\n")
    f.write("ESTADÍSTICAS DE TORQUES ORIGINALES:\n")
    f.write(f"  Min:  {min_torque:8.2f}\n")
    f.write(f"  Max:  {max_torque:8.2f}\n")
    f.write(f"  Mean: {mean_torque:8.2f}\n")
    f.write(f"  Std:  {std_torque:8.2f}\n\n")
    f.write("NORMALIZACIÓN APLICADA:\n")
    f.write(f"  norm = 2 * (real - {min_torque:.2f}) / {range_torque:.2f} - 1\n")
    f.write(f"  real = (norm + 1) * {range_torque:.2f} / 2 + {min_torque:.2f}\n")
print(f"   ✓ Configuración guardada: {config_file}")


# ========== PROCESAMIENTO DE IMÁGENES ==========
print("\n[3/5] Procesando imágenes...")

transform = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor(),
])

images_list = []
for img_path in tqdm(all_images, desc="   Procesando"):
    img = Image.open(img_path).convert('RGB')
    images_list.append(transform(img))

images_tensor = torch.stack(images_list)
num_samples   = len(all_images)

print(f"\n   ✓ Tensor de imágenes: {images_tensor.shape}")
print(f"   ✓ Tensor de torques:  {torques_normalized.shape}")


# ========== DESNORMALIZACIÓN ==========
def denormalize(norm_val, _=None):
    return (norm_val + 1) * range_torque / 2 + min_torque

print(f"\n   Desnormalización: (norm + 1) * {range_torque:.2f} / 2 + {min_torque:.2f}")


# ========== DEFINICIÓN DE ESQUIVA POR PERCENTIL (TOP 10%) ==========
torques_real_all = denormalize(torques_normalized)
abs_torque       = torch.abs(torques_real_all)

ESQUIVA_THRESHOLD = torch.quantile(abs_torque, ESQUIVA_PERCENTIL).item()

print(f"\n   Definición de esquiva por percentil:")
print(f"   Percentil: {int(ESQUIVA_PERCENTIL * 100)}")
print(f"   Umbral |torque| >= {ESQUIVA_THRESHOLD:.2f}")
print(f"   % esquivas reales: {(abs_torque >= ESQUIVA_THRESHOLD).float().mean().item() * 100:.2f}%")


# ========== SPLIT FIJO ==========
print("\n[4/5] Dividiendo datos en Train/Test...")

if os.path.exists(SPLIT_FILE):
    split     = torch.load(SPLIT_FILE)
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]
    print(f"   ✓ Usando split fijo existente: {SPLIT_FILE}")
else:
    indices    = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    train_idx  = indices[:train_size]
    test_idx   = indices[train_size:]
    torch.save({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_FILE)
    print(f"   ✓ Split creado y guardado en: {SPLIT_FILE}")

train_images  = images_tensor[train_idx].to(device)
train_torques = torques_normalized[train_idx].to(device)
test_images   = images_tensor[test_idx].to(device)
test_torques  = torques_normalized[test_idx].to(device)

print(f"   ✓ Train: {len(train_idx)} ({len(train_idx)/num_samples*100:.1f}%)")
print(f"   ✓ Test:  {len(test_idx)} ({len(test_idx)/num_samples*100:.1f}%)")


# ========== MODELO ==========
print("\n[5/5] Construyendo y entrenando modelo...")

class PilotNetMejorado(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.bn1   = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn2   = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.bn3   = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5   = nn.BatchNorm2d(64)

        self.fc1      = nn.Linear(64 * 1 * 18, 100)
        self.bn_fc1   = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2      = nn.Linear(100, 50)
        self.bn_fc2   = nn.BatchNorm1d(50)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3      = nn.Linear(50, 10)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4      = nn.Linear(10, 1)

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
print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")


# ========== LOSS: SmoothL1 (Huber) + ponderación por esquiva ==========

base_loss = nn.SmoothL1Loss(reduction="none")

def weighted_huber_loss(pred_norm_2d: torch.Tensor, target_norm_2d: torch.Tensor) -> torch.Tensor:
    """
    pred_norm_2d, target_norm_2d: shape [B,1] (normalizados)
    Etiqueta esquiva según torque real del target y aplica mayor peso.
    """
    real_t       = denormalize(target_norm_2d.squeeze(1).detach().cpu())
    esquiva_mask = (torch.abs(real_t) >= ESQUIVA_THRESHOLD).to(pred_norm_2d.device)

    weights = torch.ones_like(target_norm_2d.squeeze(1), device=pred_norm_2d.device)
    weights[esquiva_mask] = ESQUIVA_LOSS_WEIGHT

    per_sample = base_loss(pred_norm_2d.squeeze(1), target_norm_2d.squeeze(1))
    return (weights * per_sample).mean()


# ========== MÉTRICAS ==========

@torch.no_grad()
def calcular_metricas(preds_norm: torch.Tensor, reales_norm: torch.Tensor) -> dict:
    preds_t  = denormalize(preds_norm.cpu())
    reales_t = denormalize(reales_norm.cpu())

    mse_norm = torch.mean((preds_norm.cpu() - reales_norm.cpu()) ** 2).item()
    mse_t    = torch.mean((preds_t - reales_t) ** 2).item()
    rmse_t   = float(np.sqrt(mse_t))
    mae_t    = torch.mean(torch.abs(preds_t - reales_t)).item()

    den      = torch.clamp(torch.abs(reales_t), min=TORQUE_EPS)
    er       = (torch.abs(preds_t - reales_t) / den) * 100.0
    er_mean  = er.mean().item()
    er_median = er.median().item()

    sign_ok = ((preds_t > 0) == (reales_t > 0)).float().mean().item() * 100.0

    esquiva_mask = torch.abs(reales_t) >= ESQUIVA_THRESHOLD
    if esquiva_mask.any():
        mae_esq      = torch.mean(torch.abs(preds_t[esquiva_mask] - reales_t[esquiva_mask])).item()
        sign_ok_esq  = ((preds_t[esquiva_mask] > 0) == (reales_t[esquiva_mask] > 0)).float().mean().item() * 100.0
        frac_esq     = esquiva_mask.float().mean().item() * 100.0
    else:
        mae_esq     = float("nan")
        sign_ok_esq = 0.0
        frac_esq    = 0.0

    esquiva_pos_mask = reales_t >= ESQUIVA_THRESHOLD
    esquiva_neg_mask = reales_t <= -ESQUIVA_THRESHOLD

    mae_esq_pos = torch.mean(torch.abs(preds_t[esquiva_pos_mask] - reales_t[esquiva_pos_mask])).item() if esquiva_pos_mask.sum() > 0 else 0.0
    mae_esq_neg = torch.mean(torch.abs(preds_t[esquiva_neg_mask] - reales_t[esquiva_neg_mask])).item() if esquiva_neg_mask.sum() > 0 else 0.0

    return {
        "mse_norm":          mse_norm,
        "rmse_torque":       rmse_t,
        "mae_torque":        mae_t,
        "er_mean":           er_mean,
        "er_median":         er_median,
        "sign_ok":           sign_ok,
        "mae_esquiva":       mae_esq,
        "sign_ok_esquiva":   sign_ok_esq,
        "frac_esquiva_test": frac_esq,
        "mae_esquiva_pos":   mae_esq_pos,
        "mae_esquiva_neg":   mae_esq_neg,
    }


# ========== DATALOADERS ==========

train_loader = DataLoader(TensorDataset(train_images, train_torques), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(test_images,  test_torques),  batch_size=batch_size, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)


# ========== ENTRENAMIENTO ==========

train_loss_hist = []
test_mse_hist   = []
mae_hist        = []
mae_esq_hist    = []
er_med_hist     = []

best_test_mae_esq = float("inf")
best_epoch        = 0
best_metrics      = None

print("-" * 70)
print("Entrenando...")
print("-" * 70)

for epoch in range(num_epochs):
    # TRAIN
    model.train()
    train_loss = 0.0

    for imgs, torq in train_loader:
        imgs = imgs.to(device)
        torq = torq.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = weighted_huber_loss(out, torq)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)

    # TEST
    model.eval()
    test_mse_norm = 0.0
    preds_all, reals_all = [], []

    with torch.no_grad():
        for imgs, torq in test_loader:
            imgs = imgs.to(device)
            torq = torq.to(device).unsqueeze(1)
            out  = model(imgs)
            test_mse_norm += torch.mean((out - torq) ** 2).item() * imgs.size(0)
            preds_all.append(out.squeeze(1).cpu())
            reals_all.append(torq.squeeze(1).cpu())

    test_mse_norm /= len(test_loader.dataset)
    scheduler.step(test_mse_norm)

    preds = torch.cat(preds_all)
    reals = torch.cat(reals_all)
    m     = calcular_metricas(preds, reals)

    train_loss_hist.append(train_loss)
    test_mse_hist.append(test_mse_norm)
    mae_hist.append(m["mae_torque"])
    mae_esq_hist.append(m["mae_esquiva"])
    er_med_hist.append(m["er_median"])

    # Guardar mejor checkpoint por MAE en esquivas
    if not np.isnan(m["mae_esquiva"]) and m["mae_esquiva"] < best_test_mae_esq:
        best_test_mae_esq = m["mae_esquiva"]
        best_epoch        = epoch + 1
        best_metrics      = m

        torch.save({
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "normalization_params": normalization_params,
            "max_torque_ref":      max_torque_ref,
            "min_torque_ref":      min_torque_ref,
            "split_file":          SPLIT_FILE,
            "loss":                "SmoothL1(weighted)",
            "esquiva_threshold":   ESQUIVA_THRESHOLD,
            "esquiva_percentil":   ESQUIVA_PERCENTIL,
            "esquiva_loss_weight": ESQUIVA_LOSS_WEIGHT,
            "best_mae_esquiva":    best_test_mae_esq,
            "best_metrics":        best_metrics
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
            f"SignOK_esq: {m['sign_ok_esquiva']:.1f}% | "
            f"MAE_pos: {m['mae_esquiva_pos']:.2f} | "
            f"MAE_neg: {m['mae_esquiva_neg']:.2f}"
        )

print("-" * 70)
print(f"✓ Mejor modelo (por MAE esquivas) en epoch {best_epoch} | MAE_esq: {best_test_mae_esq:.2f}")
print(f"  Checkpoint: {CKPT_OUT}")
print("-" * 70)

if best_metrics is not None:
    print("\nMétricas del mejor checkpoint:")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"  {k:22s}: {v:.4f}")
        else:
            print(f"  {k:22s}: {v}")

print(f"MAE esquivas positivas: {best_metrics['mae_esquiva_pos']:.2f}")
print(f"MAE esquivas negativas: {best_metrics['mae_esquiva_neg']:.2f}")


# ========== VISUALIZACIÓN ==========

fig = plt.figure(figsize=(18, 14))

ax1 = plt.subplot(3, 2, 1)
plt.plot(train_loss_hist)
plt.xlabel("Epoch"); plt.ylabel("Train Loss (SmoothL1 weighted)")
plt.title("Train Loss"); plt.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 2, 2)
plt.plot(test_mse_hist)
plt.xlabel("Epoch"); plt.ylabel("Test MSE (norm)")
plt.title("Test MSE (norm)"); plt.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 2, 3)
plt.plot(mae_hist,     label="MAE global")
plt.plot(mae_esq_hist, label="MAE esquivas")
plt.xlabel("Epoch"); plt.ylabel("MAE (torque)")
plt.title("MAE global vs esquivas"); plt.legend(); plt.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 2, 4)
plt.plot(er_med_hist)
plt.xlabel("Epoch"); plt.ylabel("Error relativo (mediana) %")
plt.title("Error relativo (mediana)"); plt.grid(True, alpha=0.3)

# Cargar mejor modelo para los scatter
checkpoint_final = torch.load(CKPT_OUT, map_location=device)
model.load_state_dict(checkpoint_final["model_state_dict"])
model.eval()
with torch.no_grad():
    test_preds_final = model(test_images).squeeze().cpu()
test_reales_final  = test_torques.squeeze().cpu()
test_preds_torque  = denormalize(test_preds_final)
test_reales_torque = denormalize(test_reales_final)
test_esq_mask      = torch.abs(test_reales_torque) >= ESQUIVA_THRESHOLD

plt.subplot(3, 2, 5)
plt.scatter(test_reales_torque.numpy(), test_preds_torque.numpy(), alpha=0.3, s=10)
plt.plot([min_torque_ref, max_torque_ref], [min_torque_ref, max_torque_ref], 'r--', linewidth=2)
plt.xlabel("Torque Real"); plt.ylabel("Torque Predicho")
plt.title("Predicciones vs Reales (Todos)"); plt.grid(True, alpha=0.3); plt.axis('equal')

plt.subplot(3, 2, 6)
esq_r = test_reales_torque[test_esq_mask]
esq_p = test_preds_torque[test_esq_mask]
if len(esq_r) > 0:
    plt.scatter(esq_r.numpy(), esq_p.numpy(), alpha=0.5, s=15, color='red')
    lim = [min(esq_r.min(), esq_p.min()), max(esq_r.max(), esq_p.max())]
    plt.plot(lim, lim, 'k--', linewidth=2)
plt.xlabel("Torque Real (esquivas)"); plt.ylabel("Torque Predicho (esquivas)")
plt.title(f"Predicciones vs Reales (Esquivas, n={test_esq_mask.sum().item()})")
plt.grid(True, alpha=0.3); plt.axis('equal')

plt.tight_layout()
plt.savefig("pilotnet_loss_modificada.png", dpi=150)
print("\n✓ Figura guardada: pilotnet_loss_modificada.png")
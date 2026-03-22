# Importación de librerías necesarias
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

# Carpeta raíz que contiene las subcarpetas run1, run2, run3, run4
base_data_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\Dataset"
output_dir    = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_processed"

# Nombres de las subcarpetas de cada toma de datos
run_folders = ["run1", "run2", "run3", "run4"]

num_epochs    = 40
batch_size    = 64
learning_rate = 1e-3
weight_decay  = 1e-5
DEBUG         = False
TORQUE_EPS    = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("PREPROCESAMIENTO DE IMÁGENES PARA PILOTNET")
print("Estructura: Dataset/runN/telemetry_data/ + video_data/")
print("=" * 70)
print(f"Dispositivo: {device}")


# ========== CARGA DE RUTAS Y TORQUES ==========
all_images  = []
all_torques = []

print("\n[1/7] Cargando rutas de imágenes desde carpetas...")

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
        print(f"   ⚠ No se encontró la carpeta video_data en {run_path}, omitiendo {run_name}")
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
    print("\n ERROR: No se encontraron imágenes. Revisa las rutas y la estructura de carpetas.")
    exit(1)


# ========== ANÁLISIS DE DISTRIBUCIÓN DE TORQUES ==========
print("\n[2/7] Analizando distribución de torques...")
torques_tensor = torch.tensor(all_torques, dtype=torch.float32)

idx_min = torch.argmin(torques_tensor).item()
idx_max = torch.argmax(torques_tensor).item()

print(f"El torque mínimo ({torques_tensor.min().item()}) está en: {all_images[idx_min]}")
print(f"El torque máximo ({torques_tensor.max().item()}) está en: {all_images[idx_max]}")

min_torque  = torques_tensor.min().item()
max_torque  = torques_tensor.max().item()
mean_torque = torques_tensor.mean().item()
std_torque  = torques_tensor.std().item()

# Estas dos variables sustituyen a max_torque_ref y min_torque_ref del pilotnet original
max_torque_ref = max_torque
min_torque_ref = min_torque

print(f"\n Estadísticas originales:")
print(f"   Min:  {min_torque:8.2f}")
print(f"   Max:  {max_torque:8.2f}")
print(f"   Mean: {mean_torque:8.2f}")
print(f"   Std:  {std_torque:8.2f}")

num_positivos = (torques_tensor > 0).sum().item()
num_negativos = (torques_tensor < 0).sum().item()
num_ceros     = (torques_tensor == 0).sum().item()

print(f"\n Distribución de signos:")
print(f"   Positivos (derecha): {num_positivos:6d} ({num_positivos/len(all_torques)*100:.1f}%)")
print(f"   Negativos (izq):     {num_negativos:6d} ({num_negativos/len(all_torques)*100:.1f}%)")
print(f"   Ceros (recto):       {num_ceros:6d} ({num_ceros/len(all_torques)*100:.1f}%)")


# ========== NORMALIZACIÓN ==========
print("\n[3/7] Normalizando torques...")

range_torque       = max_torque - min_torque
torques_normalized = 2 * (torques_tensor - min_torque) / range_torque - 1

normalization_params = {
    'method':     'minmax_symmetric',
    'min_torque': min_torque,
    'max_torque': max_torque,
    'range':      range_torque
}

print(f"   Rango normalizado: [{torques_normalized.min():.3f}, {torques_normalized.max():.3f}]")
print(f"   Balance: {abs(torques_normalized.max() + torques_normalized.min()):.6f} (debe ser ≈0)")


# ========== PROCESAMIENTO DE IMÁGENES ==========
print("\n[4/7] Procesando imágenes...")

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

# Guardar configuración en .txt (ligero, sin tensores)
config_file = os.path.join(output_dir, "preprocessing_config_v3.txt")
with open(config_file, "w") as f:
    f.write("CONFIGURACIÓN DEL PREPROCESAMIENTO (VERSIÓN COMBINADA - TODOS LOS RUNS)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Directorio origen: {base_data_dir}\n")
    f.write(f"Runs incluidos: {', '.join(run_folders)}\n")
    f.write(f"Total de imágenes: {len(all_images)}\n")
    f.write(f"Dimensiones imagen: {list(images_tensor.shape[1:])}\n\n")
    f.write("ESTADÍSTICAS DE TORQUES ORIGINALES:\n")
    f.write(f"  Min:  {min_torque:8.2f}\n")
    f.write(f"  Max:  {max_torque:8.2f}\n")
    f.write(f"  Mean: {mean_torque:8.2f}\n")
    f.write(f"  Std:  {std_torque:8.2f}\n\n")
    f.write("DISTRIBUCIÓN DE SIGNOS:\n")
    f.write(f"  Positivos: {num_positivos} ({num_positivos/len(all_torques)*100:.1f}%)\n")
    f.write(f"  Negativos: {num_negativos} ({num_negativos/len(all_torques)*100:.1f}%)\n")
    f.write(f"  Ceros:     {num_ceros} ({num_ceros/len(all_torques)*100:.1f}%)\n\n")
    f.write("NORMALIZACIÓN APLICADA:\n")
    f.write(f"  Método: Min-Max Simétrica [-1, 1]\n")
    f.write(f"  Rango normalizado: [{torques_normalized.min():.3f}, {torques_normalized.max():.3f}]\n\n")
    f.write("FÓRMULAS DE CONVERSIÓN:\n")
    f.write(f"  norm = 2 * (real - {min_torque:.2f}) / {range_torque:.2f} - 1\n")
    f.write(f"  real = (norm + 1) * {range_torque:.2f} / 2 + {min_torque:.2f}\n")
print(f"   ✓ Configuración guardada: {config_file}")


# ========== DESNORMALIZACIÓN ==========
def denormalize(norm_val, _=None):
    return (norm_val + 1) * range_torque / 2 + min_torque

print(f"\n   Usando desnormalización Min-Max simétrica")
print(f"   Fórmula: (norm + 1) * {range_torque:.2f} / 2 + {min_torque:.2f}")

# Análisis de torques tras normalización
torques_real = denormalize(torques_normalized)
abs_torque   = torch.abs(torques_real)
ESQUIVA_THRESHOLD = torch.quantile(abs_torque, 0.90).item()

print(f"\n   Rango real:  [{torques_real.min():.2f}°, {torques_real.max():.2f}°]")
print(f"   Mean: {torques_real.mean():.2f}° | Std: {torques_real.std():.2f}°")
print(f"   Umbral esquivas (percentil 90): |torque| >= {ESQUIVA_THRESHOLD:.2f}°")


# ========== DIVISIÓN TRAIN/TEST ==========
print("\n[5/7] Dividiendo datos en Train/Test...")
SPLIT_FILE = "split_indices.pt"

if os.path.exists(SPLIT_FILE):
    split     = torch.load(SPLIT_FILE)
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]
else:
    indices    = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    train_idx  = indices[:train_size]
    test_idx   = indices[train_size:]
    torch.save({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_FILE)

train_images  = images_tensor[train_idx]
train_torques = torques_normalized[train_idx]
test_images   = images_tensor[test_idx]
test_torques  = torques_normalized[test_idx]

print(f"   ✓ Train: {len(train_idx)} ({len(train_idx)/num_samples*100:.1f}%)")
print(f"   ✓ Test:  {len(test_idx)} ({len(test_idx)/num_samples*100:.1f}%)")


# ========== MODELO ==========
print("\n[6/7] Construyendo y entrenando modelo PilotNet...")

class PilotNetMejorado(nn.Module):
    def __init__(self):
        super(PilotNetMejorado, self).__init__()

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
        x = self.fc4(x)  # Sin activación para preservar signo
        return x

model = PilotNetMejorado().to(device)
print(f"   ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")


# ========== MÉTRICAS ==========
def calcular_metricas(preds_norm, reales_norm):
    preds_torque  = denormalize(preds_norm)
    reales_torque = denormalize(reales_norm)

    mse_norm  = torch.mean((preds_norm - reales_norm) ** 2).item()
    mse_torque = torch.mean((preds_torque - reales_torque) ** 2).item()
    mae_torque = torch.mean(torch.abs(preds_torque - reales_torque)).item()

    den       = torch.clamp(torch.abs(reales_torque), min=TORQUE_EPS)
    error_rel = torch.abs(preds_torque - reales_torque) / den
    er_mean   = (error_rel * 100).mean().item()
    er_median = (error_rel * 100).median().item()

    esquiva_mask = torch.abs(reales_torque) >= ESQUIVA_THRESHOLD
    if esquiva_mask.sum() > 0:
        mae_esquiva = torch.abs(
            preds_torque[esquiva_mask] - reales_torque[esquiva_mask]
        ).mean().item()
        porcentaje_signo_esquiva = (
            (preds_torque[esquiva_mask] > 0) == (reales_torque[esquiva_mask] > 0)
        ).float().mean().item() * 100
    else:
        mae_esquiva = 0.0
        porcentaje_signo_esquiva = 0.0

    return {
        'mse_norm': mse_norm,
        'mse_torque': mse_torque,
        'mae_torque': mae_torque,
        'er_mean': er_mean,
        'er_median': er_median,
        'rmse_torque': np.sqrt(mse_torque),
        'mae_esquiva': mae_esquiva,
        'porcentaje_signo_esquiva': porcentaje_signo_esquiva
    }


# ========== ENTRENAMIENTO ==========
train_loader = DataLoader(
    TensorDataset(train_images, train_torques),
    batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(test_images, test_torques),
    batch_size=batch_size, shuffle=False
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

train_losses_norm      = []
test_losses_norm       = []
train_metrics_history  = []
test_metrics_history   = []
best_test_er_median    = float('inf')
best_epoch             = 0

print("-" * 70)

for epoch in range(num_epochs):
    # TRAIN
    model.train()
    train_loss_norm = 0.0
    train_preds_all = []
    train_reales_all = []

    for imgs, torques in train_loader:
        imgs, torques = imgs.to(device), torques.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, torques)

        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"NaN/Inf en outputs en epoch {epoch+1}")
            break
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf en loss en epoch {epoch+1}")
            break

        loss.backward()
        optimizer.step()
        train_loss_norm += loss.item() * imgs.size(0)
        train_preds_all.append(outputs.detach().cpu())
        train_reales_all.append(torques.detach().cpu())

    train_loss_norm /= len(train_loader.dataset)
    train_metrics = calcular_metricas(torch.cat(train_preds_all), torch.cat(train_reales_all))

    # TEST
    model.eval()
    test_loss_norm  = 0.0
    test_preds_all  = []
    test_reales_all = []

    with torch.no_grad():
        for imgs, torques in test_loader:
            imgs, torques = imgs.to(device), torques.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, torques)
            test_loss_norm += loss.item() * imgs.size(0)
            test_preds_all.append(outputs.cpu())
            test_reales_all.append(torques.cpu())

    test_loss_norm /= len(test_loader.dataset)
    test_metrics = calcular_metricas(torch.cat(test_preds_all), torch.cat(test_reales_all))

    train_losses_norm.append(train_loss_norm)
    test_losses_norm.append(test_loss_norm)
    train_metrics_history.append(train_metrics)
    test_metrics_history.append(test_metrics)

    scheduler.step(test_loss_norm)

    # Guardar mejor modelo
    if test_metrics['er_median'] < best_test_er_median:
        best_test_er_median = test_metrics['er_median']
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'normalization_params': normalization_params,
            'max_torque_ref': max_torque_ref,
            'min_torque_ref': min_torque_ref,
            'test_er_median': best_test_er_median,
        }, "pilotnet_weights.pth")

    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss: {test_loss_norm:.6f} | "
              f"MAE: {test_metrics['mae_torque']:6.2f}° | "
              f"ER_median: {test_metrics['er_median']:5.1f}%")

print("-" * 70)
print(f"✓ Mejor modelo en época {best_epoch} con ER = {best_test_er_median:.2f}%")


# ========== ANÁLISIS FINAL ==========
print("\n[7/7] Análisis final y visualizaciones...")

checkpoint = torch.load("pilotnet_weights.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    test_preds  = model(test_images).squeeze()
    test_reales = test_torques.squeeze()

final_metrics     = calcular_metricas(test_preds, test_reales)
test_preds_torque  = denormalize(test_preds.cpu())
test_reales_torque = denormalize(test_reales.cpu())

print("\n" + "=" * 70)
print("MÉTRICAS FINALES")
print("=" * 70)
print(f"MSE (norm):     {final_metrics['mse_norm']:.6f}")
print(f"RMSE (torque):  {final_metrics['rmse_torque']:.2f}°")
print(f"MAE (torque):   {final_metrics['mae_torque']:.2f}°")
print(f"ER (mean):      {final_metrics['er_mean']:.2f}%")
print(f"ER (median):    {final_metrics['er_median']:.2f}%")
print(f"MAE esquivas:   {final_metrics['mae_esquiva']:.2f}°")
print(f"Signo esquivas: {final_metrics['porcentaje_signo_esquiva']:.1f}%")
signos_correctos = ((test_preds_torque > 0) == (test_reales_torque > 0)).sum().item()
print(f"Signo correcto: {signos_correctos}/{len(test_preds_torque)} ({signos_correctos/len(test_preds_torque)*100:.1f}%)")
print("=" * 70)


# ========== VISUALIZACIÓN ==========
fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 4, 1)
plt.plot(train_losses_norm, label="Train")
plt.plot(test_losses_norm, label="Test")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss (norm)")
plt.title("Loss Durante Entrenamiento"); plt.legend(); plt.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 4, 2)
plt.plot([m['mae_torque'] for m in train_metrics_history], label="Train")
plt.plot([m['mae_torque'] for m in test_metrics_history], label="Test")
plt.xlabel("Epoch"); plt.ylabel("MAE")
plt.title("Mean Absolute Error"); plt.legend(); plt.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 4, 3)
plt.plot([m['er_median'] for m in train_metrics_history], label="Train")
plt.plot([m['er_median'] for m in test_metrics_history], label="Test")
plt.xlabel("Epoch"); plt.ylabel("Error Relativo Mediana (%)")
plt.title("Error Relativo Mediana (Métrica Principal)"); plt.legend(); plt.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 4, 4)
plt.scatter(test_reales_torque.numpy(), test_preds_torque.numpy(), alpha=0.3, s=10)
plt.plot([min_torque_ref, max_torque_ref], [min_torque_ref, max_torque_ref], 'r--', linewidth=2)
plt.xlabel("Torque Real"); plt.ylabel("Torque Predicho")
plt.title("Predicciones vs Reales"); plt.grid(True, alpha=0.3); plt.axis('equal')

ax5 = plt.subplot(2, 4, 5)
test_esquiva_mask = torch.abs(test_reales_torque) >= ESQUIVA_THRESHOLD
esquivas_reales   = test_reales_torque[test_esquiva_mask]
esquivas_preds    = test_preds_torque[test_esquiva_mask]
if len(esquivas_reales) > 0:
    plt.scatter(esquivas_reales.numpy(), esquivas_preds.numpy(), alpha=0.5, s=15, color='red')
    min_esq = min(esquivas_reales.min(), esquivas_preds.min())
    max_esq = max(esquivas_reales.max(), esquivas_preds.max())
    plt.plot([min_esq, max_esq], [min_esq, max_esq], 'k--', linewidth=2)
plt.xlabel("Torque Real (esquivas)"); plt.ylabel("Torque Predicho (esquivas)")
plt.title(f"Predicciones vs Reales (Esquivas, n={test_esquiva_mask.sum().item()})")
plt.grid(True, alpha=0.3); plt.axis('equal')

ax6 = plt.subplot(2, 4, 6)
errors = torch.abs(test_preds_torque - test_reales_torque)
plt.hist(errors.numpy(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("Error Absoluto"); plt.ylabel("Frecuencia")
plt.title("Distribución de Errores"); plt.grid(True, alpha=0.3)

ax7 = plt.subplot(2, 4, 7)
n = min(100, len(test_reales_torque))
plt.plot(range(n), test_reales_torque[:n].numpy(), 'b-', label="Real", alpha=0.7, linewidth=2)
plt.plot(range(n), test_preds_torque[:n].numpy(),  'r--', label="Predicho", alpha=0.7, linewidth=2)
plt.xlabel("Muestra"); plt.ylabel("Torque")
plt.title(f"Primeros {n} Ejemplos"); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pilotnet_weights.png", dpi=150)
print(f"   ✓ Gráfica: pilotnet_weights.png")

# Guardar modelo final
torch.save({
    'model_state_dict': model.state_dict(),
    'normalization_params': normalization_params,
    'max_torque_ref': max_torque_ref,
    'min_torque_ref': min_torque_ref,
    'train_losses': train_losses_norm,
    'test_losses': test_losses_norm,
    'train_metrics': train_metrics_history,
    'test_metrics': test_metrics_history,
    'final_metrics': final_metrics,
    'best_epoch': best_epoch,
}, "pilotnet_weights.pth")

print(f"   ✓ Modelo: pilotnet_weights.pth")

print("\n" + "=" * 70)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print(f"Mejor época:     {best_epoch}")
print(f"ER Median final: {final_metrics['er_median']:.2f}%")
print(f"MAE final:       {final_metrics['mae_torque']:.2f}°")
print("=" * 70)
"""
pilotnet_federated.py

Implementación de Federated Learning adaptada al proyecto PilotNet.

Estructura esperada:
    Dataset/
    ├── run1/
    │   ├── telemetry_data/frame-torque.txt
    │   └── video_data/frame_videos/  (imágenes .jpg)
    ├── run2/ ...
    ├── run3/ ...
    └── run4/ ...

Flujo:
    1. Carga imágenes y torques de cada run por separado (4 subconjuntos reales)
    2. Normaliza con Min-Max simétrica calculada sobre todos los datos
    3. Entrena 4 modelos individuales (uno por run)
    4. Entrena 1 modelo federado (FedAvg por epoch sobre los 4 runs)
    5. Carga el modelo combinado ya entrenado como referencia centralizada
    6. Genera las 3 gráficas: Training Loss, MSI, MSE Performance Matrix

Salidas:
    training_loss_comparison.png
    msi_comparison.png
    mse_performance_matrix.png
    cross_subset_generalization.png
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURACIÓN — ajusta estas rutas a tu entorno
# ============================================================

DATASET_DIR = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\Dataset"

# Nombres de las carpetas de cada run dentro de DATASET_DIR
RUN_NAMES = ["run1", "run2", "run3", "run4"]

# Hiperparámetros de entrenamiento
NUM_EPOCHS   = 40       # igual que en tus otros modelos
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-5

# Porcentaje de cada run que se reserva para test compartido
TEST_FRACTION = 0.20

MAX_SAMPLES_PER_RUN = 6000

torch.manual_seed(626)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PILOTNET - FEDERATED LEARNING (4 runs reales)")
print("=" * 70)
print(f"Dispositivo: {device}")
print(f"Runs: {RUN_NAMES}")
print(f"Epochs: {NUM_EPOCHS}")


# ============================================================
# ARQUITECTURA — idéntica a PilotNet de tus scripts
# ============================================================

class PilotNet(nn.Module):
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
        return self.fc4(x)


# ============================================================
# CARGA DE DATOS POR RUN
# ============================================================

img_transform = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor(),
])

def load_run(run_name):
    run_dir    = os.path.join(DATASET_DIR, run_name)
    telemetry_folder = "telemetry_data" if os.path.isdir(os.path.join(run_dir, "telemetry_data")) else "telemetry_csv"
    txt_path   = os.path.join(run_dir, telemetry_folder, "frame-torque.txt")
    frames_dir = os.path.join(run_dir, "video_data", "frame_videos")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"No encontrado: {txt_path}")
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"No encontrado: {frames_dir}")

    # Leer torques
    torque_map = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname  = os.path.basename(parts[0])
            torque = float(parts[1].split(',')[0])
            torque_map[fname] = torque

    # Obtener lista de filenames válidos (con torque disponible)
    fnames = sorted([
        fn for fn in os.listdir(frames_dir)
        if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
        and fn in torque_map
    ])

    # Submuestreo ANTES de cargar imágenes — así nunca se llena la RAM
    if MAX_SAMPLES_PER_RUN is not None and len(fnames) > MAX_SAMPLES_PER_RUN:
        idx    = torch.randperm(len(fnames))[:MAX_SAMPLES_PER_RUN].tolist()
        fnames = [fnames[i] for i in sorted(idx)]  # mantener orden temporal
        print(f"  [{run_name}] Submuestreado a {MAX_SAMPLES_PER_RUN} muestras antes de cargar")

    # Cargar solo las imágenes seleccionadas
    images_list  = []
    torques_list = []
    skipped = 0

    for fname in fnames:
        img_path = os.path.join(frames_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img_transform(img)
            images_list.append(img)
            torques_list.append(torque_map[fname])
        except Exception:
            skipped += 1
            continue

    if skipped > 0:
        print(f"  [{run_name}] Advertencia: {skipped} imágenes corruptas omitidas")

    images_tensor  = torch.stack(images_list)
    torques_tensor = torch.tensor(torques_list, dtype=torch.float32)

    print(f"  [{run_name}] {len(torques_tensor)} muestras cargadas")
    return images_tensor, torques_tensor


print("\n[1/7] Cargando y guardando datos de cada run en disco...")

CACHE_DIR = os.path.join(os.path.dirname(DATASET_DIR), "federated_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

raw_stats = []  # para calcular normalización global

for rn in RUN_NAMES:
    cache_path = os.path.join(CACHE_DIR, f"{rn}_raw.pt")
    if os.path.exists(cache_path):
        print(f"  [{rn}] Cache encontrado, omitiendo carga...")
        torqs = torch.load(cache_path)["torques"]
    else:
        imgs, torqs = load_run(rn)
        torch.save({"images": imgs, "torques": torqs}, cache_path)
        del imgs  # liberar RAM inmediatamente
    raw_stats.append(torqs)

print("\n[2/7] Normalización global Min-Max simétrica...")
all_torques_raw = torch.cat(raw_stats)
min_torque   = all_torques_raw.min().item()
max_torque   = all_torques_raw.max().item()
range_torque = max_torque - min_torque
del all_torques_raw, raw_stats

print(f"  Rango real: [{min_torque:.2f}°, {max_torque:.2f}°]")
print(f"  Rango total: {range_torque:.2f}°")

def normalize(t):  return (t - min_torque) / range_torque * 2.0 - 1.0
def denormalize(t): return (t + 1.0) * range_torque / 2.0 + min_torque

print(f"\n[3/7] Dividiendo train/test (20% test por run)...")

run_train = {}
run_test  = {}
test_images_list  = []
test_torques_list = []

for rn in RUN_NAMES:
    cache_path = os.path.join(CACHE_DIR, f"{rn}_raw.pt")
    data = torch.load(cache_path)
    imgs  = data["images"]
    torqs = normalize(data["torques"])
    del data

    n       = len(torqs)
    idx     = torch.randperm(n)
    test_n  = max(1, int(n * TEST_FRACTION))
    train_n = n - test_n

    train_idx = idx[:train_n]
    test_idx  = idx[train_n:]

    run_train[rn] = (imgs[train_idx], torqs[train_idx])
    run_test[rn]  = (imgs[test_idx],  torqs[test_idx])

    test_images_list.append(imgs[test_idx])
    test_torques_list.append(torqs[test_idx])

    del imgs, torqs
    print(f"  [{rn}] train={train_n} | test={test_n}")

test_images_global  = torch.cat(test_images_list)
test_torques_global = torch.cat(test_torques_list)
del test_images_list, test_torques_list
print(f"  Test global: {len(test_torques_global)} muestras")

# DataLoaders de entrenamiento por run
run_loaders = {}
for rn in RUN_NAMES:
    imgs, torqs = run_train[rn]
    ds = TensorDataset(imgs.to(device), torqs.to(device))
    run_loaders[rn] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(
    TensorDataset(test_images_global, test_torques_global),
    batch_size=BATCH_SIZE, shuffle=False
)


# ============================================================
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

criterion = nn.MSELoss()

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for imgs, torqs in loader:
        imgs  = imgs.to(device)
        torqs = torqs.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, torqs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, torqs in loader:
            imgs  = imgs.to(device)
            torqs = torqs.to(device).unsqueeze(1)
            out   = model(imgs)
            loss  = criterion(out, torqs)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def federated_averaging(models):
    """Promedia los pesos de todos los modelos (FedAvg)."""
    avg_state = {}
    for key in models[0].state_dict().keys():
        avg_state[key] = torch.stack(
            [m.state_dict()[key].float() for m in models]
        ).mean(0)
    return avg_state


# ============================================================
# ENTRENAMIENTO DE LOS 4 MODELOS INDIVIDUALES
# ============================================================

print(f"\n[4/7] Entrenando modelos individuales ({NUM_EPOCHS} epochs cada uno)...")

individual_models   = {}
individual_losses   = {}
individual_optimizers = {}

for rn in RUN_NAMES:
    model = PilotNet().to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    individual_models[rn]    = model
    individual_optimizers[rn] = opt
    individual_losses[rn]    = []

for epoch in range(NUM_EPOCHS):
    for rn in RUN_NAMES:
        loss = train_one_epoch(
            individual_models[rn],
            run_loaders[rn],
            individual_optimizers[rn]
        )
        individual_losses[rn].append(loss)

    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
        losses_str = " | ".join(
            f"{rn}: {individual_losses[rn][-1]:.4f}" for rn in RUN_NAMES
        )
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | {losses_str}")

# Test loss de cada modelo individual
individual_test_losses = {}
for rn in RUN_NAMES:
    individual_test_losses[rn] = evaluate(individual_models[rn], test_loader)
    print(f"  [{rn}] Test Loss: {individual_test_losses[rn]:.6f}")


# ============================================================
# ENTRENAMIENTO DEL MODELO FEDERADO (FedAvg por epoch)
# ============================================================

print(f"\n[5/7] Entrenando modelo federado ({NUM_EPOCHS} epochs)...")

fed_models = [PilotNet().to(device) for _ in RUN_NAMES]
fed_optimizers = [
    optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for m in fed_models
]
fed_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    for i, rn in enumerate(RUN_NAMES):
        loss = train_one_epoch(fed_models[i], run_loaders[rn], fed_optimizers[i])
        epoch_losses.append(loss)

    # FedAvg: promediar pesos al final de cada epoch
    avg_state = federated_averaging(fed_models)
    for m in fed_models:
        m.load_state_dict(avg_state)

    fed_losses.append(np.mean(epoch_losses))

    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
        print(f"  Federated Epoch {epoch+1:3d}/{NUM_EPOCHS} | Avg Loss: {fed_losses[-1]:.4f}")

fed_test_loss = evaluate(fed_models[0], test_loader)
print(f"  Federated Test Loss: {fed_test_loss:.6f}")


# ============================================================
# MODELO CENTRALIZADO — entrenado con las 4 tomas unificadas
# ============================================================

print(f"\n[6/7] Entrenando modelo centralizado (las 4 tomas unificadas)...")

# Concatenar datos de entrenamiento de los 4 runs
all_train_imgs_central  = torch.cat([run_train[rn][0] for rn in RUN_NAMES]).to(device)
all_train_torqs_central = torch.cat([run_train[rn][1] for rn in RUN_NAMES]).to(device)

central_loader = DataLoader(
    TensorDataset(all_train_imgs_central, all_train_torqs_central),
    batch_size=BATCH_SIZE, shuffle=True
)

centralized_model = PilotNet().to(device)
central_optimizer = optim.Adam(centralized_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
centralized_losses = []

for epoch in range(NUM_EPOCHS):
    loss = train_one_epoch(centralized_model, central_loader, central_optimizer)
    centralized_losses.append(loss)

    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
        print(f"  Centralizado Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {loss:.4f}")

centralized_test_loss = evaluate(centralized_model, test_loader)
print(f"  Centralizado Test Loss: {centralized_test_loss:.6f}")

# ============================================================
# GRÁFICA 1 — Training Loss vs Epoch
# ============================================================

plt.figure(figsize=(12, 7))
colors_runs = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for i, rn in enumerate(RUN_NAMES):
    plt.plot(individual_losses[rn], label=f'Modelo {i+1} ({rn})',
             alpha=0.75, linewidth=1.5, color=colors_runs[i])

plt.plot(fed_losses, label='Modelo Federado',
         linewidth=2.5, color='#CC0000', alpha=0.9)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss (MSE norm)', fontsize=12)
plt.title('Training Loss vs Epoch — Comparación todos los escenarios\n(Federated Averaging por epoch)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Anotar valores finales
for i, rn in enumerate(RUN_NAMES):
    plt.annotate(
        f'Modelo {i+1}: {individual_losses[rn][-1]:.4f}',
        xy=(NUM_EPOCHS - 1, individual_losses[rn][-1]),
        xytext=(NUM_EPOCHS * 0.6, individual_losses[rn][-1]),
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
    )
plt.annotate(
    f'Federado: {fed_losses[-1]:.4f}',
    xy=(NUM_EPOCHS - 1, fed_losses[-1]),
    xytext=(NUM_EPOCHS * 0.6, fed_losses[-1]),
    fontsize=8,
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
)

plt.tight_layout()
plt.savefig("training_loss_comparison.png", dpi=150)
plt.close()
print("\n[OK] training_loss_comparison.png guardado")


# ============================================================
# MSI — Model Stability Index
# ============================================================

def calculate_msi(model, subsets):
    """
    Evalúa el modelo en cada subconjunto y calcula el MSI.
    MSI = std(losses) / mean(losses)  — menor es más estable.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for imgs, torqs in subsets:
            if len(imgs) == 0:
                continue
            loader = DataLoader(
                TensorDataset(imgs.to(device), torqs.to(device)),
                batch_size=BATCH_SIZE, shuffle=False
            )
            total, n = 0.0, 0
            for bi, bt in loader:
                out   = model(bi)
                loss  = criterion(out, bt.unsqueeze(1))
                total += loss.item() * bi.size(0)
                n     += bi.size(0)
            losses.append(total / n)
    msi = np.std(losses) / np.mean(losses) if len(losses) > 1 else 0.0
    return msi, losses

# Subconjuntos de evaluación: datos de test de cada run
eval_subsets = [(run_test[rn][0], run_test[rn][1]) for rn in RUN_NAMES]

msi_individual = {}
for rn in RUN_NAMES:
    msi_individual[rn], _ = calculate_msi(individual_models[rn], eval_subsets)

msi_fed, _    = calculate_msi(fed_models[0], eval_subsets)
msi_central, _ = calculate_msi(centralized_model, eval_subsets)

# Gráfica MSI
labels_msi  = [f'Modelo {i+1}\n({rn})' for i, rn in enumerate(RUN_NAMES)] + ['Modelo\nFederado', 'Modelo\nCentralizado']
msi_values  = [msi_individual[rn] for rn in RUN_NAMES] + [msi_fed, msi_central]
test_losses_all = [individual_test_losses[rn] for rn in RUN_NAMES] + [fed_test_loss, centralized_test_loss]
colors_msi  = ['#FF6B6B', '#FF9F43', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

plt.figure(figsize=(13, 7))
bars = plt.bar(labels_msi, msi_values, color=colors_msi[:len(labels_msi)],
               alpha=0.85, edgecolor='black', linewidth=1)

for i, (bar, msi) in enumerate(zip(bars, msi_values)):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f'{msi:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.text(bar.get_x() + bar.get_width() / 2, -0.012,
             f'MSE: {test_losses_all[i]:.4f}', ha='center', va='top',
             fontsize=9, style='italic', color='darkblue')

plt.ylabel('Model Stability Index (MSI)', fontsize=13)
plt.title('Model Stability Index — Comparación entre escenarios\n(Federated Averaging por epoch)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.subplots_adjust(bottom=0.18)
plt.figtext(0.5, 0.03,
            'MSI = Desviación estándar / Media de losses entre subconjuntos\n'
            'Un MSI menor indica mayor estabilidad entre distintas distribuciones de datos',
            ha='center', fontsize=9, style='italic')
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("msi_comparison.png", dpi=150)
plt.close()
print("[OK] msi_comparison.png guardado")


# ============================================================
# GRÁFICA 3 — MSE Performance Matrix (heatmap)
# ============================================================

def eval_on_subset(model, imgs, torqs):
    model.eval()
    loader = DataLoader(
        TensorDataset(imgs.to(device), torqs.to(device)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    total, n = 0.0, 0
    with torch.no_grad():
        for bi, bt in loader:
            out  = model(bi)
            loss = criterion(out, bt.unsqueeze(1))
            total += loss.item() * bi.size(0)
            n     += bi.size(0)
    return total / n if n > 0 else float('inf')

# Subconjuntos de evaluación cruzada
all_train_imgs  = torch.cat([run_train[rn][0] for rn in RUN_NAMES])
all_train_torqs = torch.cat([run_train[rn][1] for rn in RUN_NAMES])

eval_sets = {rn: run_test[rn] for rn in RUN_NAMES}
eval_sets['Test Global'] = (test_images_global.cpu(), test_torques_global.cpu())
eval_sets['Train Completo'] = (all_train_imgs, all_train_torqs)

models_eval = {f'Modelo {i+1} ({rn})': individual_models[rn] for i, rn in enumerate(RUN_NAMES)}
models_eval['Federado']     = fed_models[0]
models_eval['Centralizado'] = centralized_model

model_names_eval  = list(models_eval.keys())
subset_names_eval = list(eval_sets.keys())

heatmap_data = []
for mn in model_names_eval:
    row = []
    for sn in subset_names_eval:
        imgs, torqs = eval_sets[sn]
        row.append(eval_on_subset(models_eval[mn], imgs, torqs))
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

plt.figure(figsize=(14, 7))
im = plt.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
plt.colorbar(im, label='MSE')
plt.xticks(range(len(subset_names_eval)), subset_names_eval, rotation=30, ha='right', fontsize=10)
plt.yticks(range(len(model_names_eval)), model_names_eval, fontsize=10)
for i in range(len(model_names_eval)):
    for j in range(len(subset_names_eval)):
        plt.text(j, i, f'{heatmap_data[i, j]:.4f}',
                 ha='center', va='center', color='black', fontweight='bold', fontsize=9)
plt.title('MSE Performance Matrix: Modelos vs Subconjuntos de evaluación\n(Valores menores son mejores — Federated Averaging por epoch)',
          fontsize=13, fontweight='bold')
plt.xlabel('Conjunto de evaluación', fontsize=11)
plt.ylabel('Modelo', fontsize=11)
plt.tight_layout()
plt.savefig("mse_performance_matrix.png", dpi=150)
plt.close()
print("[OK] mse_performance_matrix.png guardado")


# ============================================================
# GRÁFICA 4 — Cross-Subset Generalization
# ============================================================

avg_mses, std_mses, test_mses_bar = [], [], []
for mn in model_names_eval:
    run_mses = [eval_on_subset(models_eval[mn], run_test[rn][0], run_test[rn][1]) for rn in RUN_NAMES]
    avg_mses.append(np.mean(run_mses))
    std_mses.append(np.std(run_mses))
    test_mses_bar.append(eval_on_subset(models_eval[mn], test_images_global.cpu(), test_torques_global.cpu()))

x = np.arange(len(model_names_eval))
width = 0.35

plt.figure(figsize=(13, 7))
b1 = plt.bar(x - width / 2, avg_mses, width, label='MSE medio entre runs', alpha=0.8,
             color='skyblue', edgecolor='black')
b2 = plt.bar(x + width / 2, test_mses_bar, width, label='MSE test global', alpha=0.8,
             color='lightcoral', edgecolor='black')
plt.errorbar(x - width / 2, avg_mses, yerr=std_mses, fmt='none', color='black', capsize=5)

for i, (bar1, bar2) in enumerate(zip(b1, b2)):
    plt.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() + 0.0002,
             f'{avg_mses[i]:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() + 0.0002,
             f'{test_mses_bar[i]:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.xlabel('Modelo', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.title('Generalización entre subconjuntos\n(Barras de error = desviación estándar entre runs — Federated Averaging por epoch)',
          fontsize=13, fontweight='bold')
plt.xticks(x, model_names_eval, rotation=30, ha='right', fontsize=9)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("cross_subset_generalization.png", dpi=150)
plt.close()
print("[OK] cross_subset_generalization.png guardado")


# ============================================================
# RESUMEN FINAL EN CONSOLA
# ============================================================

print("\n" + "=" * 70)
print("RESUMEN FEDERATED LEARNING")
print("=" * 70)
for i, rn in enumerate(RUN_NAMES):
    print(f"  Modelo {i+1} ({rn})  — Test Loss: {individual_test_losses[rn]:.6f} | MSI: {msi_individual[rn]:.4f}")
print(f"  Modelo Federado    — Test Loss: {fed_test_loss:.6f} | MSI: {msi_fed:.4f}")
print(f"  Modelo Centralizado— Test Loss: {centralized_test_loss:.6f} | MSI: {msi_central:.4f}")
print()

fed_idx     = len(RUN_NAMES)
central_idx = len(RUN_NAMES) + 1
mejora_test = (test_mses_bar[central_idx] - test_mses_bar[fed_idx]) / test_mses_bar[central_idx] * 100
mejora_gen  = (avg_mses[central_idx] - avg_mses[fed_idx]) / avg_mses[central_idx] * 100
print(f"  Federado vs Centralizado — Test: {mejora_test:+.2f}% | Generalización: {mejora_gen:+.2f}%")
print("=" * 70)
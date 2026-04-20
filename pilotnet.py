# Importación de librerías necesarias
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# ========== CONFIGURACIÓN INICIAL ==========
torch.manual_seed(626)
processed_data_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_processed"

DATA_FILE = "processed_data.pt"

num_epochs = 40
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-5
DEBUG = False
TORQUE_EPS = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("ENTRENAMIENTO PILOTNET - VERSIÓN UNIVERSAL")
print("=" * 70)
print(f"Dispositivo: {device}")
print(f"Archivo de datos: {DATA_FILE}")


# ========== FUNCIONES DE DESNORMALIZACIÓN ==========
def create_denormalize_function(normalization_params):
    """
    Crea función de desnormalización según el método usado.
    
    Args:
        normalization_params: Dict con parámetros de normalización
    
    Returns:
        Función para desnormalizar
    """
    if normalization_params is None:
        # Normalización antigua (por max_abs)
        def denorm(norm_val, max_val):
            return norm_val * max_val
        return denorm
    
    method = normalization_params.get('method', 'unknown')
    
    if method == 'minmax_symmetric':
        # Normalización nueva (min-max simétrica)
        min_torque = normalization_params['min_torque']
        max_torque = normalization_params['max_torque']
        range_torque = normalization_params['range']
        
        def denorm(norm_val, _=None):
            return (norm_val + 1) * range_torque / 2 + min_torque
        
        print(f"\n  Usando desnormalización Min-Max simétrica")
        print(f"   Fórmula: (norm + 1) * {range_torque:.2f} / 2 + {min_torque:.2f}")
        return denorm
    else:
        # Por defecto, asumir normalización por max_abs
        def denorm(norm_val, max_val):
            return norm_val * max_val
        return denorm


# ========== CARGA DE DATOS ==========
print("\n[1/7] Cargando datos preprocesados...")
data_file = os.path.join(processed_data_dir, DATA_FILE)

if not os.path.exists(data_file):
    print(f"\n ERROR: No se encontró el archivo '{data_file}'")
    print("   Archivos disponibles:")
    for f in os.listdir(processed_data_dir):
        if f.endswith('.pt'):
            print(f"   - {f}")
    exit(1)

# Cargar datos
data = torch.load(data_file)
images_tensor = data['images']
torques_tensor = data['torques']
num_samples = data['num_samples']

# Detectar tipo de normalización
if 'normalization_params' in data:
    # Nueva versión con parámetros explícitos
    normalization_params = data['normalization_params']
    denormalize = create_denormalize_function(normalization_params)
    max_torque_ref = data['original_stats']['max']
    min_torque_ref = data['original_stats']['min']
else:
    # Versión antigua con max_torque simple
    normalization_params = None
    max_torque_ref = data['max_torque']
    min_torque_ref = data.get('min_torque', -max_torque_ref)
    denormalize = create_denormalize_function(None)

print(f"   ✓ Datos cargados exitosamente")
print(f"   ✓ Imágenes: {images_tensor.shape}")
print(f"   ✓ Torques normalizados: {torques_tensor.shape}")
print(f"   ✓ Rango real: [{min_torque_ref:.2f}°, {max_torque_ref:.2f}°]")

# Análisis de distribución
torques_real = denormalize(torques_tensor, max_torque_ref)

print(f"\n  Análisis de torques:")
print(f"   Normalizado:")
print(f"      Min: {torques_tensor.min():.4f} | Max: {torques_tensor.max():.4f}")
print(f"      Mean: {torques_tensor.mean():.4f} | Std: {torques_tensor.std():.4f}")
print(f"\n   Escala real (torque):")
print(f"      Min: {torques_real.min():.2f}° | Max: {torques_real.max():.2f}°")
print(f"      Mean: {torques_real.mean():.2f}° | Std: {torques_real.std():.2f}°")

pos_count = (torques_real > 0).sum().item()
neg_count = (torques_real < 0).sum().item()
print(f"\n   Distribución:")
print(f"      Positivos: {pos_count} ({pos_count/num_samples*100:.1f}%)")
print(f"      Negativos: {neg_count} ({neg_count/num_samples*100:.1f}%)")

abs_torque = torch.abs(torques_real)
ESQUIVA_THRESHOLD = torch.quantile(abs_torque, 0.90).item()
print(f"\n   Umbral esquivas (percentil 90): |torque| >= {ESQUIVA_THRESHOLD:.2f}°")


# ========== DIVISIÓN TRAIN/TEST ==========
print("\n[2/7] Dividiendo datos en Train/Test...")
SPLIT_FILE = "split_indices.pt"

if os.path.exists(SPLIT_FILE):
    split = torch.load(SPLIT_FILE)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
else:
    indices = torch.randperm(num_samples)
    train_size = int(0.8 * num_samples)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    torch.save({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_FILE)

train_images = images_tensor[train_idx]
train_torques = torques_tensor[train_idx]
test_images = images_tensor[test_idx]
test_torques = torques_tensor[test_idx]

print(f"   ✓ Train: {len(train_idx)} ({len(train_idx)/num_samples*100:.1f}%)")
print(f"   ✓ Test: {len(test_idx)} ({len(test_idx)/num_samples*100:.1f}%)")


# ========== MODELO ==========
print("\n[3/7] Construyendo modelo PilotNet mejorado...")

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        
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
        x = self.fc4(x)  # Sin activación para preservar signo
        
        return x

model = PilotNet().to(device)
print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")


# ========== MÉTRICAS ==========
def calcular_metricas(preds_norm, reales_norm):
    """Calcula métricas en ambas escalas."""
    # Desnormalizar
    preds_torque = denormalize(preds_norm, max_torque_ref)
    reales_torque = denormalize(reales_norm, max_torque_ref)
    
    # MSE normalizado
    mse_norm = torch.mean((preds_norm - reales_norm) ** 2).item()
    
    # Métricas en grados
    mse_torque = torch.mean((preds_torque - reales_torque) ** 2).item()
    mae_torque = torch.mean(torch.abs(preds_torque - reales_torque)).item()
    
    # Error relativo
    den = torch.clamp(torch.abs(reales_torque), min=TORQUE_EPS)
    error_rel = torch.abs(preds_torque - reales_torque) / den
    er_mean = (error_rel * 100).mean().item()
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


# ========== PREPARACIÓN ==========
print("\n[4/7] Preparando entrenamiento...")

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

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

print(f"   ✓ Configuración lista")


# ========== ENTRENAMIENTO ==========
print("\n[5/7] Entrenando modelo...")
print("-" * 70)

train_losses_norm = []
test_losses_norm = []
train_metrics_history = []
test_metrics_history = []
best_test_er_median = float('inf')
best_epoch = 0

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
    train_preds_tensor = torch.cat(train_preds_all)
    train_reales_tensor = torch.cat(train_reales_all)
    train_metrics = calcular_metricas(train_preds_tensor, train_reales_tensor)

    # TEST
    model.eval()
    test_loss_norm = 0.0
    test_preds_all = []
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
    test_preds_tensor = torch.cat(test_preds_all)
    test_reales_tensor = torch.cat(test_reales_all)
    test_metrics = calcular_metricas(test_preds_tensor, test_reales_tensor)

    # Guardar
    train_losses_norm.append(train_loss_norm)
    test_losses_norm.append(test_loss_norm)
    train_metrics_history.append(train_metrics)
    test_metrics_history.append(test_metrics)
    
    scheduler.step(test_loss_norm)
    
    # Mejor modelo
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
            'data_file': DATA_FILE
        }, "pilotnet_weights.pth")
    
    # Mostrar progreso
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss: {test_loss_norm:.6f} | "
              f"MAE: {test_metrics['mae_torque']:6.2f}° | "
              f"ER_median: {test_metrics['er_median']:5.1f}%")

print("-" * 70)
print(f"✓ Mejor modelo en época {best_epoch} con ER = {best_test_er_median:.2f}%")


# ========== ANÁLISIS FINAL ==========
print("\n[6/7] Análisis final...")

checkpoint = torch.load("pilotnet_weights.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    test_preds = model(test_images).squeeze()
    test_reales = test_torques.squeeze()

final_metrics = calcular_metricas(test_preds, test_reales)
print(f"MAE esquivas:   {final_metrics['mae_esquiva']:.2f}°")
print(f"Signo esquivas: {final_metrics['porcentaje_signo_esquiva']:.1f}%")

test_preds_torque = denormalize(test_preds.cpu(), max_torque_ref)
test_reales_torque = denormalize(test_reales.cpu(), max_torque_ref)

print("\n" + "=" * 70)
print("MÉTRICAS FINALES")
print("=" * 70)
print(f"MSE (norm):     {final_metrics['mse_norm']:.6f}")
print(f"RMSE (torque):  {final_metrics['rmse_torque']:.2f}°")
print(f"MAE (torque):   {final_metrics['mae_torque']:.2f}°")
print(f"ER (mean):      {final_metrics['er_mean']:.2f}%")
print(f"ER (median):    {final_metrics['er_median']:.2f}%")

signos_correctos = ((test_preds_torque > 0) == (test_reales_torque > 0)).sum().item()
print(f"Signo correcto: {signos_correctos}/{len(test_preds_torque)} ({signos_correctos/len(test_preds_torque)*100:.1f}%)")
print("=" * 70)


# ========== VISUALIZACIÓN ==========
print("\n[7/7] Generando visualizaciones...")

fig = plt.figure(figsize=(20, 12))

# 1. Loss
ax1 = plt.subplot(2, 4, 1)
plt.plot(train_losses_norm, label="Train")
plt.plot(test_losses_norm, label="Test")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss (norm)")
plt.title("Loss Durante Entrenamiento")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. MAE
ax2 = plt.subplot(2, 4, 2)
mae_train = [m['mae_torque'] for m in train_metrics_history]
mae_test = [m['mae_torque'] for m in test_metrics_history]
plt.plot(mae_train, label="Train")
plt.plot(mae_test, label="Test")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Mean Absolute Error")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Error Relativo Mediana
ax3 = plt.subplot(2, 4, 3)
er_train = [m['er_median'] for m in train_metrics_history]
er_test = [m['er_median'] for m in test_metrics_history]
plt.plot(er_train, label="Train")
plt.plot(er_test, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Error Relativo Mediana (%)")
plt.title("Error Relativo Mediana (Métrica Principal)")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Scatter
ax4 = plt.subplot(2, 4, 4)
plt.scatter(test_reales_torque.numpy(), test_preds_torque.numpy(), alpha=0.3, s=10)
plt.plot([min_torque_ref, max_torque_ref], [min_torque_ref, max_torque_ref], 'r--', linewidth=2)
plt.xlabel("Torque Real")
plt.ylabel("Torque Predicho")
plt.title("Predicciones vs Reales")
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 5. Scatter esquivas
ax_esq = plt.subplot(2, 4, 5)
test_esquiva_mask = torch.abs(test_reales_torque) >= ESQUIVA_THRESHOLD
esquivas_reales = test_reales_torque[test_esquiva_mask]
esquivas_preds  = test_preds_torque[test_esquiva_mask]
if len(esquivas_reales) > 0:
    plt.scatter(esquivas_reales.numpy(), esquivas_preds.numpy(), alpha=0.5, s=15, color='red')
    min_esq = min(esquivas_reales.min(), esquivas_preds.min())
    max_esq = max(esquivas_reales.max(), esquivas_preds.max())
    plt.plot([min_esq, max_esq], [min_esq, max_esq], 'k--', linewidth=2)
plt.xlabel("Torque Real (esquivas)")
plt.ylabel("Torque Predicho (esquivas)")
plt.title(f"Predicciones vs Reales (Esquivas, n={test_esquiva_mask.sum().item()})")
plt.grid(True, alpha=0.3)
plt.axis('equal')

# 6. Distribución de errores
ax5 = plt.subplot(2, 4, 6)
errors = torch.abs(test_preds_torque - test_reales_torque)
plt.hist(errors.numpy(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel("Error Absoluto")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores")
plt.grid(True, alpha=0.3)

# 7. Ejemplos
ax6 = plt.subplot(2, 4, 7)
n = min(100, len(test_reales_torque))
x = range(n)
plt.plot(x, test_reales_torque[:n].numpy(), 'b-', label="Real", alpha=0.7, linewidth=2)
plt.plot(x, test_preds_torque[:n].numpy(), 'r--', label="Predicho", alpha=0.7, linewidth=2)
plt.xlabel("Muestra")
plt.ylabel("Torque")
plt.title(f"Primeros {n} Ejemplos")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pilotnet_weights.png", dpi=150)
print(f"    Gráfica: pilotnet_weights.png")

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
    'data_file': DATA_FILE
}, "pilotnet_weights.pth")

print(f"   Modelo: pilotnet_weights.pth")

print("\n" + "=" * 70)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 70)
print(f"Mejor época: {best_epoch}")
print(f"ER Median final: {final_metrics['er_median']:.2f}%")
print(f"MAE final: {final_metrics['mae_torque']:.2f}°")
print("=" * 70)
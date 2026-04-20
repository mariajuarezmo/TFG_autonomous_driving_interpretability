"""
GRID SEARCH OPTIMIZADO CON FINE-TUNING DESDE PILOTNET
=======================================================

Código adaptado a tu pilotnet.py existente.
Usa pesos preentrenados para ahorrar ~75% del tiempo.

TIEMPOS:
- Entrenamiento completo (40 épocas × 81 modelos): 18-24 horas
- Fine-tuning desde Baseline (10 épocas × 81 modelos): 4-6 horas ✓
"""

import os
import torch
import PIL
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import itertools
import time


# CONFIGURACIÓN - ADAPTAR A TUS DIRECTORIOS

torch.manual_seed(626)

# Directorio raíz con las subcarpetas run1, run2, run3, run4
base_data_dir = r"./Dataset"
run_folders   = ["run1", "run2", "run3", "run4"]

PILOTNET_WEIGHTS_FILE = "./pilotnet_processed/pilotnet_weights.pth"
OUTPUT_DIR = "./grid_search_results"

# Hiperparámetros
EPOCHS_FINETUNING = 10
BATCH_SIZE        = 64
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-5
TORQUE_EPS        = 5.0

# Grid Search
PESOS_POSITIVOS = np.arange(1.0, 5.5, 0.5)  # [1.0, 1.5, ..., 5.0]
PESOS_NEGATIVOS = np.arange(1.0, 5.5, 0.5)

OUTPUT_DIR = "./grid_search_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("GRID SEARCH OPTIMIZADO CON FINE-TUNING ")
print("=" * 80)
print(f"Dispositivo:       {device}")
print(f"Pesos Baseline:    {PILOTNET_WEIGHTS_FILE}")
print(f"Directorio datos:  {base_data_dir}")
print(f"Épocas fine-tuning: {EPOCHS_FINETUNING}")
print(f"Total combinaciones: {len(PESOS_POSITIVOS) * len(PESOS_NEGATIVOS)}")
print("=" * 80)


# PREPROCESAMIENTO EN MEMORIA (extraído de preproceso_pilotnet_dataset.py)

def preprocess_images_in_memory():
    """
    Carga y preprocesa todas las imágenes en memoria RAM/VRAM.
    Equivalente a preproceso_pilotnet_dataset.py pero SIN guardar en disco.
    Devuelve images_tensor, torques_normalized y normalization_params.
    """
    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO EN MEMORIA")
    print("=" * 70)


    # 1. Cargar rutas e imágenes

    all_images  = []
    all_torques = []

    print("\n[1/3] Cargando rutas de imágenes")
    for run_name in run_folders:
        run_path = os.path.join(base_data_dir, run_name)
        if not os.path.isdir(run_path):
            print(f"    Carpeta no encontrada, omitiendo: {run_path}")
            continue

        telemetry_dir    = os.path.join(run_path, "telemetry_data")
        video_dir        = os.path.join(run_path, "video_data", "frame_videos")
        frame_torque_txt = os.path.join(telemetry_dir, "frame-torque.txt")

        if not os.path.exists(frame_torque_txt):
            print(f"    No se encontró frame-torque.txt en {telemetry_dir}, omitiendo {run_name}")
            continue
        if not os.path.isdir(video_dir):
            print(f"    No se encontró video_data en {run_path}, omitiendo {run_name}")
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

        print(f"    {run_name}: {len(all_images) - count_before} imágenes cargadas")

    print(f"\n    Total imágenes encontradas: {len(all_images)}")
    if len(all_images) == 0:
        raise RuntimeError("No se encontraron imágenes. Revisa las rutas.")


    # 2. Normalización Min-Max simétrica [-1, 1]

    print("\n[2/3] Normalizando torques (Min-Max simétrica)")
    torques_tensor = torch.tensor(all_torques, dtype=torch.float32)

    min_torque  = torques_tensor.min().item()
    max_torque  = torques_tensor.max().item()
    mean_torque = torques_tensor.mean().item()
    std_torque  = torques_tensor.std().item()
    range_torque = max_torque - min_torque

    torques_normalized = 2 * (torques_tensor - min_torque) / range_torque - 1

    normalization_params = {
        'method':     'minmax_symmetric',
        'min_torque': min_torque,
        'max_torque': max_torque,
        'range':      range_torque,
    }

    print(f"   Min real:  {min_torque:.2f}  →  normalizado: {torques_normalized.min():.3f}")
    print(f"   Max real:  {max_torque:.2f}  →  normalizado: {torques_normalized.max():.3f}")
    print(f"   Media:     {mean_torque:.2f}  |  Std: {std_torque:.2f}")


    # 3. Procesar imágenes

    print("\n[3/3] Procesando imágenes (CenterCrop 66×200 + ToTensor)")

    transform = transforms.Compose([
        transforms.CenterCrop((66, 200)),
        transforms.ToTensor(),
    ])

    images_list       = []
    valid_torques_list = []

    for i, img_path in enumerate(tqdm(all_images, desc="   Procesando")):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images_list.append(img)
            valid_torques_list.append(torques_normalized[i])
        except (UnidentifiedImageError, OSError):
            print(f"\n    Saltando imagen dañada: {img_path}")
            continue

    images_tensor      = torch.stack(images_list)
    torques_normalized = torch.stack(valid_torques_list)

    print(f"\n    Imágenes válidas:   {len(images_tensor)}")
    print(f"    Shape imágenes:     {images_tensor.shape}")
    print(f"    Shape torques:      {torques_normalized.shape}")
    print("=" * 70)

    return images_tensor, torques_normalized, normalization_params, {
        'min': min_torque, 'max': max_torque,
        'mean': mean_torque, 'std': std_torque
    }


# FUNCIONES DE DESNORMALIZACIÓN

def create_denormalize_function(normalization_params):
    if normalization_params is None:
        def denorm(norm_val, max_val):
            return norm_val * max_val
        return denorm

    method = normalization_params.get('method', 'unknown')
    if method == 'minmax_symmetric':
        min_torque   = normalization_params['min_torque']
        range_torque = normalization_params['range']
        def denorm(norm_val, _=None):
            return (norm_val + 1) * range_torque / 2 + min_torque
        return denorm
    else:
        def denorm(norm_val, max_val):
            return norm_val * max_val
        return denorm


# MODELO PILOTNET

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1  = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.bn1    = nn.BatchNorm2d(24)
        self.conv2  = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.bn2    = nn.BatchNorm2d(36)
        self.conv3  = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.bn3    = nn.BatchNorm2d(48)
        self.conv4  = nn.Conv2d(48, 64, kernel_size=3)
        self.bn4    = nn.BatchNorm2d(64)
        self.conv5  = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5    = nn.BatchNorm2d(64)
        self.fc1    = nn.Linear(64 * 1 * 18, 100)
        self.bn_fc1 = nn.BatchNorm1d(100)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2    = nn.Linear(100, 50)
        self.bn_fc2 = nn.BatchNorm1d(50)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3    = nn.Linear(50, 10)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4    = nn.Linear(10, 1)

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
        x = self.fc4(x)
        return x


# CARGA Y SPLIT (usa datos ya preprocesados en memoria)

def prepare_data_split(images_tensor, torques_normalized, normalization_params, original_stats):
    num_samples    = len(images_tensor)
    denormalize    = create_denormalize_function(normalization_params)
    max_torque_ref = original_stats['max']
    min_torque_ref = original_stats['min']

    # Split 80/20 — reutiliza índices guardados si existen
    SPLIT_FILE = "split_indices.pt"
    if os.path.exists(SPLIT_FILE):
        split     = torch.load(SPLIT_FILE)
        train_idx = split["train_idx"]
        test_idx  = split["test_idx"]
        print(f"   Split cargado desde {SPLIT_FILE}")
    else:
        indices    = torch.randperm(num_samples)
        train_size = int(0.8 * num_samples)
        train_idx  = indices[:train_size]
        test_idx   = indices[train_size:]
        torch.save({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_FILE)
        print(f"   Split generado y guardado en {SPLIT_FILE}")

    train_images  = images_tensor[train_idx]
    train_torques = torques_normalized[train_idx]
    test_images   = images_tensor[test_idx]
    test_torques  = torques_normalized[test_idx]

    print(f"   Total: {num_samples}  |  Train: {len(train_idx)}  |  Test: {len(test_idx)}")

    # Umbral de esquivas (percentil 90 del torque absoluto)
    torques_real      = denormalize(torques_normalized, max_torque_ref)
    abs_torque        = torch.abs(torques_real)
    esquiva_threshold = torch.quantile(abs_torque, 0.90).item()

    train_torques_real   = denormalize(train_torques, max_torque_ref)
    esquivas_pos_mask    = (train_torques_real >= esquiva_threshold) & (train_torques_real > 0)
    esquivas_neg_mask    = (torch.abs(train_torques_real) >= esquiva_threshold) & (train_torques_real < 0)
    esquivas_pos_idx     = torch.where(esquivas_pos_mask)[0]
    esquivas_neg_idx     = torch.where(esquivas_neg_mask)[0]

    print(f"   Umbral esquivas (P90): {esquiva_threshold:.2f}°")
    print(f"   Esquivas positivas:    {len(esquivas_pos_idx)}")
    print(f"   Esquivas negativas:    {len(esquivas_neg_idx)}")

    train_dataset = TensorDataset(train_images, train_torques)
    test_dataset  = TensorDataset(test_images,  test_torques)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return {
        'train_loader':      train_loader,
        'test_loader':       test_loader,
        'train_dataset':     train_dataset,
        'test_dataset':      test_dataset,
        'esquivas_pos_idx':  esquivas_pos_idx,
        'esquivas_neg_idx':  esquivas_neg_idx,
        'denormalize':       denormalize,
        'max_torque_ref':    max_torque_ref,
        'min_torque_ref':    min_torque_ref,
        'normalization_params': normalization_params,
        'esquiva_threshold': esquiva_threshold,
    }


# CARGA DE PESOS PILOTNET

def load_baseline_weights(model, weights_file):
    print(f"\n[PILOTNET] Cargando pesos: {weights_file}")
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"No se encontró {weights_file}")
    
    # Añadimos weights_only=False para evitar el aviso
    checkpoint = torch.load(weights_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   Pesos cargados exitosamente")
    return model, checkpoint

# MÉTRICAS

def calcular_metricas_simple(preds_norm, reales_norm, denormalize_fn,
                              max_torque_ref, esquiva_threshold, esquivas_mask):
    preds_torque  = denormalize_fn(preds_norm,  max_torque_ref)
    reales_torque = denormalize_fn(reales_norm, max_torque_ref)
    mae_global   = torch.mean(torch.abs(preds_torque - reales_torque)).item()
    mae_esquivas = torch.mean(torch.abs(preds_torque[esquivas_mask] - reales_torque[esquivas_mask])).item()
    return mae_global, mae_esquivas


# FINE-TUNING

def train_model_with_finetuning(pos_weight, neg_weight, data_dict, num_epochs=EPOCHS_FINETUNING):
    model, _ = load_baseline_weights(PilotNet().to(device), PILOTNET_WEIGHTS_FILE)

    total_train = len(data_dict['train_dataset'])
    weights     = np.ones(total_train)
    weights[data_dict['esquivas_pos_idx'].numpy()] = pos_weight
    weights[data_dict['esquivas_neg_idx'].numpy()] = neg_weight

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_loader_resampled = DataLoader(data_dict['train_dataset'], batch_size=BATCH_SIZE, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_mae_esq    = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        for imgs, torques in train_loader_resampled:
            imgs    = imgs.to(device)
            torques = torques.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss    = criterion(outputs, torques)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds_all  = []
            test_reales_all = []
            for imgs, torques in data_dict['test_loader']:
                imgs    = imgs.to(device)
                torques = torques.to(device).unsqueeze(1)
                test_preds_all.append(model(imgs).cpu())
                test_reales_all.append(torques.cpu())

        test_preds  = torch.cat(test_preds_all).squeeze()
        test_reales = torch.cat(test_reales_all).squeeze()

        test_reales_denorm = data_dict['denormalize'](test_reales, data_dict['max_torque_ref'])
        esquivas_test_mask = torch.abs(test_reales_denorm) >= data_dict['esquiva_threshold']

        mae_global, mae_esq = calcular_metricas_simple(
            test_preds, test_reales, data_dict['denormalize'],
            data_dict['max_torque_ref'], data_dict['esquiva_threshold'], esquivas_test_mask
        )

        if mae_esq < best_mae_esq:
            best_mae_esq     = mae_esq
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        final_preds_all  = []
        final_reales_all = []
        for imgs, torques in data_dict['test_loader']:
            imgs    = imgs.to(device)
            torques = torques.to(device).unsqueeze(1)
            final_preds_all.append(model(imgs).cpu())
            final_reales_all.append(torques.cpu())

    final_preds  = torch.cat(final_preds_all).squeeze()
    final_reales = torch.cat(final_reales_all).squeeze()

    final_reales_denorm = data_dict['denormalize'](final_reales, data_dict['max_torque_ref'])
    esquivas_final_mask = torch.abs(final_reales_denorm) >= data_dict['esquiva_threshold']

    mae_global, mae_esq = calcular_metricas_simple(
        final_preds, final_reales, data_dict['denormalize'],
        data_dict['max_torque_ref'], data_dict['esquiva_threshold'], esquivas_final_mask
    )

    return {'mae_global': mae_global, 'mae_esquivas': mae_esq}


# GRID SEARCH PRINCIPAL

def run_grid_search():

    # 1. Preprocesar imágenes en memoria (sin guardar .pt)
    images_tensor, torques_normalized, normalization_params, original_stats = \
        preprocess_images_in_memory()

    # 2. Preparar split y dataloaders
    print("\n[INIT] Preparando split y dataloaders")
    data_dict = prepare_data_split(
        images_tensor, torques_normalized, normalization_params, original_stats
    )

    # 3. Evaluar Baseline
    print(f"\n[PILOTNET] Evaluando modelo Baseline")
    model_baseline, _ = load_baseline_weights(PilotNet().to(device), PILOTNET_WEIGHTS_FILE)
    model_baseline.eval()

    with torch.no_grad():
        baseline_preds_all  = []
        baseline_reales_all = []
        for imgs, torques in data_dict['test_loader']:
            imgs    = imgs.to(device)
            torques = torques.to(device).unsqueeze(1)
            baseline_preds_all.append(model_baseline(imgs).cpu())
            baseline_reales_all.append(torques.cpu())

    baseline_preds  = torch.cat(baseline_preds_all).squeeze()
    baseline_reales = torch.cat(baseline_reales_all).squeeze()

    baseline_reales_denorm = data_dict['denormalize'](baseline_reales, data_dict['max_torque_ref'])
    esquivas_baseline_mask = torch.abs(baseline_reales_denorm) >= data_dict['esquiva_threshold']

    baseline_mae_global, baseline_mae_esq = calcular_metricas_simple(
        baseline_preds, baseline_reales, data_dict['denormalize'],
        data_dict['max_torque_ref'], data_dict['esquiva_threshold'], esquivas_baseline_mask
    )

    print(f"  Baseline MAE esquivas: {baseline_mae_esq:.2f}°")
    print(f"  Baseline MAE global:   {baseline_mae_global:.2f}°")

    # 4. Grid search
    print("\n" + "=" * 80)
    print("INICIANDO GRID SEARCH")
    print("=" * 80 + "\n")

    results      = []
    combinations = list(itertools.product(PESOS_POSITIVOS, PESOS_NEGATIVOS))
    start_time   = time.time()

    for idx, (pos_w, neg_w) in enumerate(combinations, 1):
        print(f"[{idx:2d}/{len(combinations)}] pos={pos_w:.1f}, neg={neg_w:.1f} ... ", end='', flush=True)

        try:
            metrics = train_model_with_finetuning(pos_w, neg_w, data_dict, EPOCHS_FINETUNING)

            mejora_pct      = ((baseline_mae_esq - metrics['mae_esquivas']) / baseline_mae_esq) * 100
            degradacion_pct = ((metrics['mae_global'] - baseline_mae_global) / baseline_mae_global) * 100
            if degradacion_pct <= 0:
                degradacion_pct = 0.01
            ratio = mejora_pct / degradacion_pct

            results.append({
                'pos_weight':       pos_w,
                'neg_weight':       neg_w,
                'mae_global':       metrics['mae_global'],
                'mae_esquivas':     metrics['mae_esquivas'],
                'mejora_pct':       mejora_pct,
                'degradacion_pct':  degradacion_pct,
                'ratio':            ratio,
            })

            # Checkpoint CSV tras cada combinación
            pd.DataFrame(results).to_csv(
                os.path.join(OUTPUT_DIR, 'grid_search_results_finetuning.csv'), index=False
            )

            print(f" MAE esq: {metrics['mae_esquivas']:.2f}° | Ratio: {ratio:.2f}x")

        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nTiempo total: {elapsed_hours:.2f} horas")

    df_results = pd.DataFrame(results)

    print("\n TOP 5 COMBINACIONES:")
    print(df_results.nlargest(5, 'ratio')[['pos_weight', 'neg_weight', 'mae_esquivas', 'ratio']].to_string(index=False))

    best = df_results.loc[df_results['ratio'].idxmax()]
    print(f"\n ÓPTIMO ENCONTRADO:")
    print(f"  pos_weight:   {best['pos_weight']}")
    print(f"  neg_weight:   {best['neg_weight']}")
    print(f"  MAE esquivas: {best['mae_esquivas']:.2f}°")
    print(f"  MAE global:   {best['mae_global']:.2f}°")
    print(f"  Mejora:       {best['mejora_pct']:.1f}%")
    print(f"  Degradación:  {best['degradacion_pct']:.1f}%")
    print(f"  Ratio:        {best['ratio']:.2f}x")

    with open(os.path.join(OUTPUT_DIR, 'OPTIMO_ENCONTRADO.txt'), 'w') as f:
        f.write(f"Óptimo encontrado:\n")
        f.write(f"pos_weight:        {best['pos_weight']}\n")
        f.write(f"neg_weight:        {best['neg_weight']}\n")
        f.write(f"Ratio:             {best['ratio']:.2f}x\n")
        f.write(f"Mejora esquivas:   {best['mejora_pct']:.1f}%\n")
        f.write(f"Degradación global:{best['degradacion_pct']:.1f}%\n")

    print(f"\n Resultados guardados en: {OUTPUT_DIR}/")
    return df_results


# MAIN
if __name__ == '__main__':
    try:
        df_results = run_grid_search()
        print("\n Grid search completado exitosamente")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
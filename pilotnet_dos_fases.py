"""
pilotnet_two_phase.py
Entrenamiento en dos fases para PilotNet:
  - Fase 1: entrenamiento general (MSE, muestreo normal)
  - Fase 2: fine-tuning centrado en esquivas (SmoothL1/Huber + pesos por esquiva)
Guarda:
  - checkpoint del mejor modelo (por MAE en esquivas) -> pilotnet_two_phase_best.pth
  - checkpoint final -> pilotnet_two_phase_final.pth
  - figura resumen -> pilotnet_two_phase.png
  - split fijo -> split_indices.pt (para comparar con otros métodos)
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


# =========================
# CONFIG
# =========================
torch.manual_seed(626)

processed_data_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_processed"
DATA_FILE = "processed_data_v3.pt"

# Fase 1 (general)
PHASE1_EPOCHS = 20
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-5

# Fase 2 (fine-tuning)
PHASE2_EPOCHS = 10
PHASE2_LR = 1e-4
PHASE2_WEIGHT_DECAY = 0.0  # a veces conviene 0 en fine-tuning

# Dataloader
BATCH_SIZE = 64

# Definición de "esquiva"
ESQUIVA_PERCENTIL = 0.90     # top 10%
ESQUIVA_THRESHOLD = None 
ESQUIVA_LOSS_WEIGHT = 3.0      # peso extra en loss para muestras esquiva

# Métrica relativa robusta
TORQUE_EPS = 5.0               # evita explosión del relativo cuando real ~ 0

# Opciones fase 2
USE_WEIGHTED_SAMPLER_PHASE2 = False  # True si quieres además muestreo ponderado en fase 2
FREEZE_CONVS_PHASE2 = False          # True si quieres congelar convs en fase 2

# Split fijo (para comparativas)
SPLIT_FILE = "split_indices.pt"

# Salidas
BEST_CKPT = "pilotnet_two_phase_best.pth"
FINAL_CKPT = "pilotnet_two_phase_final.pth"
FIG_OUT = "pilotnet_two_phase.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DESNORMALIZACIÓN
# =========================
def create_denormalize_function(normalization_params):
    """Crea función de desnormalización según método usado en el .pt."""
    if normalization_params is None:
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

    def denorm(norm_val, max_val):
        return norm_val * max_val

    return denorm


# =========================
# MODELO (igual que tu PilotNetMejorado)
# =========================
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
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(100, 50)
        self.bn_fc2 = nn.BatchNorm1d(50)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(50, 10)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = self.drop1(torch.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(torch.relu(self.bn_fc2(self.fc2(x))))
        x = self.drop3(torch.relu(self.fc3(x)))

        return self.fc4(x)  # salida lineal para preservar signo


def freeze_convs_and_bns(model: nn.Module, freeze: bool = True):
    """Congela o descongela conv+bn para fase 2."""
    for name, p in model.named_parameters():
        if name.startswith("conv") or name.startswith("bn"):
            p.requires_grad = not freeze


# =========================
# MÉTRICAS
# =========================
@torch.no_grad()
def calcular_metricas(preds_norm: torch.Tensor,
                      reales_norm: torch.Tensor,
                      denormalize_fn,
                      max_torque_ref: float,
                      esquiva_threshold: float,
                      torque_eps: float):
    """
    preds_norm / reales_norm: tensores 1D en escala normalizada
    Devuelve métricas globales y específicas de esquiva.
    """
    preds_t = denormalize_fn(preds_norm.cpu(), max_torque_ref)
    reales_t = denormalize_fn(reales_norm.cpu(), max_torque_ref)

    # MSE normalizado
    mse_norm = torch.mean((preds_norm.cpu() - reales_norm.cpu()) ** 2).item()

    # Errores en torque real
    mse_t = torch.mean((preds_t - reales_t) ** 2).item()
    rmse_t = float(np.sqrt(mse_t))
    mae_t = torch.mean(torch.abs(preds_t - reales_t)).item()

    # Error relativo robusto (evitando 0)
    den = torch.clamp(torch.abs(reales_t), min=torque_eps)
    er = (torch.abs(preds_t - reales_t) / den) * 100.0
    er_mean = er.mean().item()
    er_median = er.median().item()

    # Signo correcto
    sign_ok = ((preds_t > 0) == (reales_t > 0)).float().mean().item() * 100.0

    # Esquivas (por umbral)
    esquiva_mask = torch.abs(reales_t) >= esquiva_threshold
    if esquiva_mask.any():
        mae_esq = torch.mean(torch.abs(preds_t[esquiva_mask] - reales_t[esquiva_mask])).item()
        rmse_esq = float(torch.sqrt(torch.mean((preds_t[esquiva_mask] - reales_t[esquiva_mask]) ** 2)).item())
        sign_ok_esq = ((preds_t[esquiva_mask] > 0) == (reales_t[esquiva_mask] > 0)).float().mean().item() * 100.0
        frac_esq = esquiva_mask.float().mean().item() * 100.0
    else:
        mae_esq = float("nan")
        rmse_esq = float("nan")
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
        "rmse_esquiva": rmse_esq,
        "sign_ok_esquiva": sign_ok_esq,
        "frac_esquiva_test": frac_esq
    }


# =========================
# LOSS FASE 2 (Huber/SmoothL1 ponderada por esquiva)
# =========================
smoothl1 = nn.SmoothL1Loss(reduction="none")

def weighted_huber_loss(pred_norm_2d: torch.Tensor,
                        target_norm_2d: torch.Tensor,
                        denormalize_fn,
                        max_torque_ref: float,
                        esquiva_threshold: float,
                        esquiva_weight: float):
    """
    pred_norm_2d, target_norm_2d: shape [B,1] (normalizados)
    - detecta esquiva según target en torque real
    - aplica peso extra a muestras esquiva
    """
    # Torque real (solo para definir máscara)
    real_t = denormalize_fn(target_norm_2d.squeeze(1).detach().cpu(), max_torque_ref)
    esquiva_mask = (torch.abs(real_t) >= esquiva_threshold).to(pred_norm_2d.device)

    weights = torch.ones_like(target_norm_2d.squeeze(1), device=pred_norm_2d.device)
    weights[esquiva_mask] = esquiva_weight

    per_sample = smoothl1(pred_norm_2d.squeeze(1), target_norm_2d.squeeze(1))
    return (weights * per_sample).mean()


# =========================
# EVAL LOOP
# =========================
@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    preds_all, reals_all = [], []
    mse_loss_accum = 0.0

    for imgs, torq in loader:
        imgs = imgs.to(device)
        torq = torq.to(device).unsqueeze(1)  # [B,1]
        out = model(imgs)

        # MSE norm solo para referencia/scheduler
        mse_loss_accum += torch.mean((out - torq) ** 2).item() * imgs.size(0)

        preds_all.append(out.squeeze(1).detach().cpu())
        reals_all.append(torq.squeeze(1).detach().cpu())

    mse_loss_accum /= len(loader.dataset)
    preds = torch.cat(preds_all)
    reals = torch.cat(reals_all)
    return mse_loss_accum, preds, reals


# =========================
# MAIN
# =========================
def main():
    print("=" * 70)
    print("PILOTNET - ENTRENAMIENTO EN DOS FASES")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data file: {DATA_FILE}")
    print(f"Esquiva percentil: {int(ESQUIVA_PERCENTIL*100)}")
    print(f"Fase2 loss weight (esquiva): {ESQUIVA_LOSS_WEIGHT}")
    print(f"Use WeightedSampler (fase2): {USE_WEIGHTED_SAMPLER_PHASE2}")
    print(f"Freeze convs (fase2): {FREEZE_CONVS_PHASE2}")
    print("=" * 70)

    # ---- Load data
    data_path = os.path.join(processed_data_dir, DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No existe: {data_path}")

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

    print(f"Samples: {num_samples}")
    print(f"Torque real range: [{min_torque_ref:.2f}, {max_torque_ref:.2f}]")

    # =========================
    # DEFINICIÓN DE ESQUIVA POR PERCENTIL (TOP 10%)
    # =========================
    torques_real = denormalize(torques_tensor.detach().cpu(), max_torque_ref)
    abs_torque = torch.abs(torques_real)

    ESQUIVA_THRESHOLD = torch.quantile(abs_torque, ESQUIVA_PERCENTIL).item()

    print("\nDefinición de esquiva por percentil:")
    print(f"   Percentil: {int(ESQUIVA_PERCENTIL * 100)}")
    print(f"   Umbral |torque| >= {ESQUIVA_THRESHOLD:.2f}")

    esquiva_mask = abs_torque >= ESQUIVA_THRESHOLD
    print(f"   % esquivas reales (dataset): {esquiva_mask.float().mean().item() * 100:.2f}%\n")

    # ---- Split fijo
    if os.path.exists(SPLIT_FILE):
        split = torch.load(SPLIT_FILE)
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        print(f"Usando split fijo: {SPLIT_FILE}")
    else:
        indices = torch.randperm(num_samples)
        train_size = int(0.8 * num_samples)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        torch.save({"train_idx": train_idx, "test_idx": test_idx}, SPLIT_FILE)
        print(f"Split creado y guardado: {SPLIT_FILE}")

    train_images = images_tensor[train_idx].to(device)
    train_torques = torques_tensor[train_idx].to(device)
    test_images = images_tensor[test_idx].to(device)
    test_torques = torques_tensor[test_idx].to(device)

    # Dataloaders fase 1
    train_loader_phase1 = DataLoader(
        TensorDataset(train_images, train_torques),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_torques),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Dataloader fase 2 (opcional sampler ponderado)
    if USE_WEIGHTED_SAMPLER_PHASE2:
        # calcula pesos en TRAIN en base a torque real del target
        train_torque_real = denormalize(train_torques.detach().cpu(), max_torque_ref)
        esquiva_mask = torch.abs(train_torque_real) >= ESQUIVA_THRESHOLD

        weights = torch.ones(len(train_torques), dtype=torch.float32)
        weights[esquiva_mask] = ESQUIVA_LOSS_WEIGHT
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

        train_loader_phase2 = DataLoader(
            TensorDataset(train_images, train_torques),
            batch_size=BATCH_SIZE,
            sampler=sampler
        )
    else:
        train_loader_phase2 = DataLoader(
            TensorDataset(train_images, train_torques),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

    # ---- Model
    model = PilotNetMejorado().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Histories
    hist = {
        "phase": [],
        "epoch": [],
        "train_mse_norm": [],
        "test_mse_norm": [],
        "test_mae": [],
        "test_mae_esq": [],
        "test_er_median": [],
        "test_sign_ok": [],
        "test_sign_ok_esq": [],
    }

    best_mae_esq = float("inf")
    best_epoch_global = 0
    best_phase = "?"

    # =========================
    # FASE 1: GENERAL
    # =========================
    print("\n" + "=" * 70)
    print("FASE 1 - ENTRENAMIENTO GENERAL (MSE)")
    print("=" * 70)

    criterion1 = nn.MSELoss()
    optimizer1 = optim.Adam(model.parameters(), lr=PHASE1_LR, weight_decay=PHASE1_WEIGHT_DECAY)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode="min", factor=0.5, patience=5)

    for epoch in range(PHASE1_EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for imgs, torq in train_loader_phase1:
            imgs = imgs.to(device)
            torq = torq.to(device).unsqueeze(1)

            optimizer1.zero_grad()
            out = model(imgs)
            loss = criterion1(out, torq)
            loss.backward()
            optimizer1.step()

            train_loss_accum += loss.item() * imgs.size(0)

        train_loss_accum /= len(train_loader_phase1.dataset)

        test_mse_norm, preds, reals = eval_model(model, test_loader)
        scheduler1.step(test_mse_norm)

        m = calcular_metricas(
            preds_norm=preds,
            reales_norm=reals,
            denormalize_fn=denormalize,
            max_torque_ref=max_torque_ref,
            esquiva_threshold=ESQUIVA_THRESHOLD,
            torque_eps=TORQUE_EPS
        )

        # Guardar historia
        hist["phase"].append(1)
        hist["epoch"].append(epoch + 1)
        hist["train_mse_norm"].append(train_loss_accum)
        hist["test_mse_norm"].append(test_mse_norm)
        hist["test_mae"].append(m["mae_torque"])
        hist["test_mae_esq"].append(m["mae_esquiva"])
        hist["test_er_median"].append(m["er_median"])
        hist["test_sign_ok"].append(m["sign_ok"])
        hist["test_sign_ok_esq"].append(m["sign_ok_esquiva"])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[P1] Epoch {epoch+1:3d}/{PHASE1_EPOCHS} | "
                f"TrainMSE(norm): {train_loss_accum:.6f} | "
                f"TestMSE(norm): {test_mse_norm:.6f} | "
                f"MAE: {m['mae_torque']:.2f} | "
                f"MAE_esq: {m['mae_esquiva']:.2f} | "
                f"ER_med: {m['er_median']:.1f}% | "
                f"SignOK: {m['sign_ok']:.1f}%"
            )

        # Puedes guardar "best" ya aquí si quieres, pero lo normal es priorizar fase 2.
        # Si lo quieres, descomenta:
        # if not np.isnan(m["mae_esquiva"]) and m["mae_esquiva"] < best_mae_esq:
        #     best_mae_esq = m["mae_esquiva"]
        #     best_epoch_global = epoch + 1
        #     best_phase = "phase1"
        #     torch.save({...}, BEST_CKPT)

    # =========================
    # FASE 2: FINE-TUNING EN ESQUIVAS
    # =========================
    print("\n" + "=" * 70)
    print("FASE 2 - FINE-TUNING (SmoothL1/Huber ponderada por esquiva)")
    print("=" * 70)

    if FREEZE_CONVS_PHASE2:
        freeze_convs_and_bns(model, freeze=True)
        print("Convs+BN congeladas para fase 2.")
    else:
        freeze_convs_and_bns(model, freeze=False)

    # En fase 2 optimizamos solo parámetros requires_grad=True
    params_trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = optim.Adam(params_trainable, lr=PHASE2_LR, weight_decay=PHASE2_WEIGHT_DECAY)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode="min", factor=0.5, patience=3)

    for epoch in range(PHASE2_EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for imgs, torq in train_loader_phase2:
            imgs = imgs.to(device)
            torq = torq.to(device).unsqueeze(1)

            optimizer2.zero_grad()
            out = model(imgs)

            # Loss centrada en esquivas (Huber ponderada)
            loss = weighted_huber_loss(
                pred_norm_2d=out,
                target_norm_2d=torq,
                denormalize_fn=denormalize,
                max_torque_ref=max_torque_ref,
                esquiva_threshold=ESQUIVA_THRESHOLD,
                esquiva_weight=ESQUIVA_LOSS_WEIGHT
            )
            loss.backward()
            optimizer2.step()

            train_loss_accum += loss.item() * imgs.size(0)

        train_loss_accum /= len(train_loader_phase2.dataset)

        test_mse_norm, preds, reals = eval_model(model, test_loader)
        scheduler2.step(test_mse_norm)

        m = calcular_metricas(
            preds_norm=preds,
            reales_norm=reals,
            denormalize_fn=denormalize,
            max_torque_ref=max_torque_ref,
            esquiva_threshold=ESQUIVA_THRESHOLD,
            torque_eps=TORQUE_EPS
        )

        # Guardar historia (epoch global: suma fase1 + fase2)
        global_epoch = PHASE1_EPOCHS + (epoch + 1)
        hist["phase"].append(2)
        hist["epoch"].append(global_epoch)
        hist["train_mse_norm"].append(train_loss_accum)  # aquí realmente es "loss fase2", pero lo guardamos igual
        hist["test_mse_norm"].append(test_mse_norm)
        hist["test_mae"].append(m["mae_torque"])
        hist["test_mae_esq"].append(m["mae_esquiva"])
        hist["test_er_median"].append(m["er_median"])
        hist["test_sign_ok"].append(m["sign_ok"])
        hist["test_sign_ok_esq"].append(m["sign_ok_esquiva"])

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"[P2] Epoch {epoch+1:3d}/{PHASE2_EPOCHS} | "
                f"TrainLoss(P2): {train_loss_accum:.6f} | "
                f"TestMSE(norm): {test_mse_norm:.6f} | "
                f"MAE: {m['mae_torque']:.2f} | "
                f"MAE_esq: {m['mae_esquiva']:.2f} | "
                f"ER_med: {m['er_median']:.1f}% | "
                f"SignOK_esq: {m['sign_ok_esquiva']:.1f}%"
            )

        # Guardar mejor checkpoint por MAE en esquivas (aquí sí, que es el objetivo)
        if not np.isnan(m["mae_esquiva"]) and m["mae_esquiva"] < best_mae_esq:
            best_mae_esq = m["mae_esquiva"]
            best_epoch_global = global_epoch
            best_phase = "phase2"

            torch.save({
                "best_epoch_global": best_epoch_global,
                "best_phase": best_phase,
                "model_state_dict": model.state_dict(),
                "normalization_params": normalization_params,
                "max_torque_ref": max_torque_ref,
                "min_torque_ref": min_torque_ref,
                "data_file": DATA_FILE,
                "split_file": SPLIT_FILE,
                "phase1": {
                    "epochs": PHASE1_EPOCHS,
                    "lr": PHASE1_LR,
                    "weight_decay": PHASE1_WEIGHT_DECAY,
                    "loss": "MSE"
                },
                "phase2": {
                    "epochs": PHASE2_EPOCHS,
                    "lr": PHASE2_LR,
                    "weight_decay": PHASE2_WEIGHT_DECAY,
                    "loss": "SmoothL1(weighted)",
                    "esquiva_threshold": ESQUIVA_THRESHOLD,
                    "esquiva_percentil": ESQUIVA_PERCENTIL,
                    "esquiva_weight": ESQUIVA_LOSS_WEIGHT,
                    "use_weighted_sampler": USE_WEIGHTED_SAMPLER_PHASE2,
                    "freeze_convs": FREEZE_CONVS_PHASE2
                },
                "best_mae_esquiva": best_mae_esq,
                "last_eval_metrics": m
            }, BEST_CKPT)

    # ---- Guardar checkpoint final (último estado)
    torch.save({
        "model_state_dict": model.state_dict(),
        "normalization_params": normalization_params,
        "max_torque_ref": max_torque_ref,
        "min_torque_ref": min_torque_ref,
        "data_file": DATA_FILE,
        "split_file": SPLIT_FILE,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "esquiva_threshold": ESQUIVA_THRESHOLD,
        "esquiva_percentil": ESQUIVA_PERCENTIL,
        "esquiva_weight": ESQUIVA_LOSS_WEIGHT,
        "use_weighted_sampler_phase2": USE_WEIGHTED_SAMPLER_PHASE2,
        "freeze_convs_phase2": FREEZE_CONVS_PHASE2
    }, FINAL_CKPT)

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"Mejor checkpoint: {BEST_CKPT}")
    print(f"  Mejor epoch global: {best_epoch_global} ({best_phase})")
    print(f"  Mejor MAE esquivas: {best_mae_esq:.2f}")
    print(f"Checkpoint final: {FINAL_CKPT}")
    print("=" * 70)

    # ---- Plot resumen
    plot_summary(hist)


def plot_summary(hist):
    epochs = hist["epoch"]
    phases = hist["phase"]

    fig = plt.figure(figsize=(16, 10))

    # 1) Test MSE norm
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(epochs, hist["test_mse_norm"], label="Test MSE(norm)")
    plt.xlabel("Epoch (global)")
    plt.ylabel("MSE (norm)")
    plt.title("Test MSE (norm) - Dos fases")
    plt.grid(True, alpha=0.3)

    # 2) MAE global vs MAE esquivas
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(epochs, hist["test_mae"], label="MAE (global)")
    plt.plot(epochs, hist["test_mae_esq"], label="MAE (esquivas)")
    plt.xlabel("Epoch (global)")
    plt.ylabel("MAE (torque)")
    plt.title("MAE global vs MAE esquivas")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3) ER median
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(epochs, hist["test_er_median"], label="ER median (%)")
    plt.xlabel("Epoch (global)")
    plt.ylabel("ER median (%)")
    plt.title("Error relativo (mediana) - robusto")
    plt.grid(True, alpha=0.3)

    # 4) Signo correcto
    ax4 = plt.subplot(2, 2, 4)
    plt.plot(epochs, hist["test_sign_ok"], label="Signo OK (global)")
    plt.plot(epochs, hist["test_sign_ok_esq"], label="Signo OK (esquivas)")
    plt.xlabel("Epoch (global)")
    plt.ylabel("Signo correcto (%)")
    plt.title("Exactitud de signo")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # marca el cambio de fase
    # (se asume que fase 1 ocupa epochs 1..PHASE1_EPOCHS)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150)
    print(f"\nFigura guardada: {FIG_OUT}")


if __name__ == "__main__":
    main()

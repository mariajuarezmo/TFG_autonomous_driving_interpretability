'''
El script comparativa_subida_vs_esquivas.py tiene como finalidad evaluar cuantitativamente el comportamiento del modelo PilotNet en maniobras de esquiva, 
comparando el error de predicción del torque durante:
1.	El evento de esquiva detectado automáticamente (t_start a t_end).
2.	Un baseline local definido como los 3 segundos previos al inicio del evento, evitando que el baseline incluya tramos de conducción con otras esquivas (contaminación del baseline).
Para cada vídeo, el script realiza una sola inferencia sobre todos los frames, 
calcula métricas de error por frame (MSE, AE, RE) y luego segmenta por esquiva individual generando ficheros por evento y un resumen agregado.

'''
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

try:
    import cv2
except ImportError:
    cv2 = None

# =========================
# RUTAS 
# =========================
'''
Se definen rutas absolutas a los recursos necesarios:
•	VIDEOS_DIR: carpeta con los vídeos .ts (solo se usa para leer FPS).
•	DATASETS_DIR: carpeta con los datasets por vídeo (imágenes + data.txt).
•	CARSTATE_DIR y TURNS_DIR: carpeta donde están los CSV de esquivas por vídeo (__turns.csv) y el global _ALL_TURNS.csv.
•	MODEL_PTH: modelo entrenado (.pth) con state_dict y max_torque.
•	OUT_DIR: carpeta donde se guardan los resultados del análisis.
'''
VIDEOS_DIR   = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\videos"
DATASETS_DIR = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_datasets"
CARSTATE_DIR = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales\csvs"
TURNS_DIR    = os.path.join(CARSTATE_DIR, "_turn_analysis_out")

MODEL_PTH    = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_combinado_weights.pth"
OUT_DIR      = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\mse_out_individual"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Constantes relevantes:
•	BASELINE_WINDOW_SEC = 3.0: ventana del baseline local (3 segundos).
•	EPS_RE = 1.0: epsilon para evitar división por cero en el error relativo. Se usa 1 ya que los torques están en ciertos.
•	DEFAULT_FPS = 30.0: valor por defecto si no puede leerse el FPS real.
'''
BASELINE_WINDOW_SEC = 3.0
EPS_TIME = 1e-9
EPS_RE = 1
DEFAULT_FPS = 20.0


# PREPROCESADO
'''
Preprocesado de entrada (imágenes)
Se define un transform idéntico al usado en entrenamiento:
•	CenterCrop((66, 200)): recorte central al tamaño esperado por PilotNet.
•	ToTensor(): conversión a tensor PyTorch con normalización estándar de [0,1].
'''

img_transform = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor(),
])


# ========= DESNORMALIZACIÓN =========
def create_denormalize_function(normalization_params):
    if normalization_params is None:
        # caso antiguo (max_abs)
        def denorm(norm_val, max_val):
            return norm_val * max_val
        return denorm

    method = normalization_params.get("method", "unknown")

    if method == "minmax_symmetric":
        min_torque = normalization_params["min_torque"]
        range_torque = normalization_params["range"]

        def denorm(norm_val, _=None):
            return (norm_val + 1.0) * range_torque / 2.0 + min_torque
        return denorm

    # fallback: max_abs
    def denorm(norm_val, max_val):
        return norm_val * max_val
    return denorm


# MODELO PilotNet
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
        x = self.fc4(x)
        return x

# HELPERS

'''
Buscar el vídeo .ts y obtener FPS real
find_ts_for_video_base(video_base): intenta localizar el .ts correspondiente a un dataset.
get_video_fps(ts_path): usa OpenCV si está disponible para obtener fps. Si falla, usa DEFAULT_FPS.
Esto es importante porque el script reconstruye timestamp_sec como:timestamp_sec=(frame_idx)/fps

y necesita el FPS para alinear frames con intervalos t_start / t_end del detector de esquivas. 
'''
def find_ts_for_video_base(video_base: str) -> str | None:
    cand = os.path.join(VIDEOS_DIR, f"{video_base}.ts")
    if os.path.exists(cand):
        return cand
    matches = glob.glob(os.path.join(VIDEOS_DIR, f"{video_base}*.ts"))
    return matches[0] if matches else None

def get_video_fps(ts_path: str | None) -> float:
    if cv2 is None or ts_path is None:
        return DEFAULT_FPS
    cap = cv2.VideoCapture(ts_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0 or not np.isfinite(fps):
        return DEFAULT_FPS
    return float(fps)

'''
Lectura del dataset por vídeo (data.txt)
read_data_txt(video_folder):
•	lee cada línea de data.txt con formato: img_name torque.
•	extrae frame_idx desde el nombre de la imagen (00025.jpg → 25).
•	si no puede extraer el índice, asigna un índice incremental.
•	ordena por frame_idx para reconstruir la secuencia temporal.
Esto produce un DataFrame con columnas:
•	img_name
•	y_true_torque
•	frame_idx
'''
def read_data_txt(video_folder: str) -> pd.DataFrame:
    path = os.path.join(video_folder, "data.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_name, torque = parts[0], float(parts[1])
            base = os.path.splitext(os.path.basename(img_name))[0]
            try:
                frame_idx = int(base)
            except ValueError:
                frame_idx = None
            rows.append((img_name, torque, frame_idx))

    df = pd.DataFrame(rows, columns=["img_name", "y_true_torque", "frame_idx"])
    if df["frame_idx"].isna().any():
        df["frame_idx"] = np.arange(len(df), dtype=int)
    return df.sort_values("frame_idx").reset_index(drop=True)

def base_for_turns(video_base: str) -> str:
    for suffix in ["--qcamera", "--fcamera", "--dcamera"]:
        if video_base.endswith(suffix):
            return video_base[:-len(suffix)]
    return video_base

'''
Carga segura de CSV de esquivas
•	safe_read_csv(path): evita errores si el CSV no existe o está vacío.
•	load_turns_for_video(video_base, turns_dir):
    - intenta primero cargar el CSV específico del vídeo: <video>__turns.csv
    - si no existe, intenta filtrar el _ALL_TURNS.csv global por el campo video.
Esto permite que el script funcione aunque falte el CSV específico de un vídeo. 
'''

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 0:
            return pd.DataFrame()
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def load_turns_for_video(video_base: str, turns_dir: str) -> pd.DataFrame:
    key = base_for_turns(video_base)
    per_video = os.path.join(turns_dir, f"{key}__turns.csv")
    df = safe_read_csv(per_video)
    if not df.empty:
        return df

    all_path = os.path.join(turns_dir, "_ALL_TURNS.csv")
    df_all = safe_read_csv(all_path)
    if not df_all.empty and "video" in df_all.columns:
        df_all["video"] = df_all["video"].astype(str).str.strip()
        return df_all[df_all["video"] == key].copy()

    return pd.DataFrame()

'''
Inferencia del modelo (predicción de torque)
predict_video(model, video_folder, df, max_abs_torque):
•	recorre todos los img_name en orden temporal,
•	carga imagen, aplica img_transform,
•	ejecuta el modelo para obtener salida normalizada,
•	desnormaliza multiplicando por max_abs_torque,
•	devuelve un array con y_pred_torque por frame.
La inferencia se ejecuta una sola vez por vídeo, para no recalcular predicciones por cada evento. 
'''
@torch.no_grad()
def predict_video(model: nn.Module, video_folder: str, df: pd.DataFrame, denormalize_fn, max_torque_ref: float | None) -> np.ndarray:
    model.eval()
    preds = []
    for img_name in df["img_name"].tolist():
        img_path = os.path.join(video_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        x = img_transform(img).unsqueeze(0).to(DEVICE)

        y_norm = model(x).item()  # salida en [-1, 1] (normalizada)
        y_deg = denormalize_fn(y_norm, max_torque_ref)  # a grados reales
        preds.append(float(y_deg))

    return np.array(preds, dtype=float)


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.nanmean(x)) if len(x) else np.nan,
        "median": float(np.nanmedian(x)) if len(x) else np.nan,
        "p95": float(np.nanpercentile(x, 95)) if len(x) else np.nan,
    }

'''
Construcción de máscaras baseline/evento
Para cada evento [t0, t1]:
build_event_and_baseline_masks(...) devuelve:
•	event_mask: frames dentro de [t0, t1]
•	baseline_mask: frames dentro de [t0-3s, t0) pero excluyendo cualquier frame que pertenezca a cualquier esquiva del vídeo.
Evita que el baseline se “contamine” con otros eventos de esquiva (si hay dos esquivas cercanas o solapadas). 
'''
def build_event_and_baseline_masks(df: pd.DataFrame, turns_df: pd.DataFrame, t0: float, t1: float, baseline_sec: float):
    """
    Devuelve:
      - event_mask: frames en [t0, t1]
      - baseline_mask: frames en [t0-baseline_sec, t0) EXCLUYENDO cualquier frame que caiga dentro de
        cualquier esquiva (unión de intervalos de turns_df), para evitar contaminación.
    """
    t = df["timestamp_sec"].to_numpy(dtype=float)

    # Máscara del evento actual
    event_mask = (t >= t0) & (t <= t1)

    # Ventana baseline local
    b0 = max(0.0, t0 - baseline_sec)
    baseline_mask = (t >= b0) & (t < (t0 - EPS_TIME))

    # Excluir cualquier frame que esté dentro de CUALQUIER esquiva
    # (incluyendo el propio evento y otros)
    any_evasion_mask = np.zeros(len(df), dtype=bool)
    for _, r in turns_df.iterrows():
        a0 = float(r["t_start"])
        a1 = float(r["t_end"])
        if a1 <= a0:
            continue
        any_evasion_mask |= (t >= a0) & (t <= a1)

    baseline_mask &= ~any_evasion_mask

    return baseline_mask, event_mask, b0

'''
Plot del histograma ae y er
'''
def plot_hist_metric(base_vals, ev_vals, out_png, title, metric_label, bins=60):
    """
    Pinta histograma (densidad) comparando baseline vs evento para una métrica cualquiera.
    metric_label: texto para el eje X (p.ej. "AE (|pred-true|)" o "RE (abs err / (|true|+eps))")
    """
    base_vals = np.asarray(base_vals, dtype=float)
    ev_vals   = np.asarray(ev_vals, dtype=float)

    # Si no hay muestras en alguno de los dos, no graficamos
    if len(base_vals) == 0 or len(ev_vals) == 0:
        return

    # Filtrar NaNs/infs (importante sobre todo en RE)
    base_vals = base_vals[np.isfinite(base_vals)]
    ev_vals   = ev_vals[np.isfinite(ev_vals)]
    if len(base_vals) == 0 or len(ev_vals) == 0:
        return

    plt.figure(figsize=(10, 5))

    all_vals = np.concatenate([base_vals, ev_vals])
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))

    plt.hist(base_vals, bins=bins, range=(vmin, vmax), alpha=0.6,
             density=True, label=f"Baseline ({BASELINE_WINDOW_SEC:.0f}s antes)")
    plt.hist(ev_vals, bins=bins, range=(vmin, vmax), alpha=0.6,
             density=True, label="Esquiva (evento)")

    plt.title(title)
    plt.xlabel(metric_label)
    plt.ylabel("Densidad")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    ckpt = torch.load(MODEL_PTH, map_location=DEVICE, weights_only=False)

    model = PilotNetMejorado().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalization_params = ckpt.get("normalization_params", None)
    denormalize_fn = create_denormalize_function(normalization_params)

    # Para compatibilidad:
    # - en minmax_symmetric, max_torque_ref no se usa (puede ser None)
    # - en max_abs, sí lo necesita
    max_torque_ref = ckpt.get("max_torque_ref", ckpt.get("max_torque", None))

    print("[OK] Modelo cargado.")
    if normalization_params is not None:
        print(f"  Normalización: {normalization_params.get('method', 'unknown')}")
    else:
        print("  Normalización: max_abs (legacy)")

    for video_base in sorted(os.listdir(DATASETS_DIR)):
        video_folder = os.path.join(DATASETS_DIR, video_base)
        if not os.path.isdir(video_folder):
            continue
        if not os.path.exists(os.path.join(video_folder, "data.txt")):
            continue

        print(f"\n=== {video_base} ===")

        ts_path = find_ts_for_video_base(video_base)
        fps = get_video_fps(ts_path)
        print(f"  fps={fps:.3f} | ts={'OK' if ts_path else 'NO (fallback)'}")

        df = read_data_txt(video_folder)
        df["timestamp_sec"] = df["frame_idx"].astype(float) / float(fps)

        turns_df = load_turns_for_video(video_base, TURNS_DIR).copy()
        turns_df["t_start"] = pd.to_numeric(turns_df.get("t_start"), errors="coerce")
        turns_df["t_end"]   = pd.to_numeric(turns_df.get("t_end"), errors="coerce")
        turns_df["evasion_id"] = pd.to_numeric(turns_df.get("evasion_id"), errors="coerce")
        turns_df = turns_df.dropna(subset=["t_start", "t_end"])
        turns_df = turns_df[turns_df["t_end"] > turns_df["t_start"]]

        # predicción + MSE (una vez por vídeo)
        df["y_pred_torque"] = predict_video(model, video_folder, df, denormalize_fn, max_torque_ref)
        df["mse"] = (df["y_pred_torque"] - df["y_true_torque"]) ** 2
        df["ae"] = np.abs(df["y_pred_torque"] - df["y_true_torque"])
        df["re"] = df["ae"] / (np.abs(df["y_true_torque"]) + EPS_RE)


        # Guardar CSV base del vídeo (todas las muestras)
        out_base_csv = os.path.join(OUT_DIR, f"{video_base}__base_predictions.csv")
        df.to_csv(out_base_csv, index=False)
        print(f"  [OK] Base CSV: {out_base_csv}")

                # Si no hay esquivas, no hacemos análisis por evento
        if turns_df.empty:
            print("  [INFO] No hay esquivas detectadas. (Solo subida)")
            continue

        # =========================
        # ANÁLISIS POR ESQUIVA (baseline local 3s)
        # =========================
        summary_rows = []

        # orden por evasion_id si existe; si no por t_start
        if "evasion_id" in turns_df.columns and turns_df["evasion_id"].notna().any():
            turns_df = turns_df.sort_values(["evasion_id", "t_start"])
        else:
            turns_df = turns_df.sort_values(["t_start"])

        for _, r in turns_df.iterrows():
            ev_id = int(r["evasion_id"]) if ("evasion_id" in turns_df.columns and pd.notna(r.get("evasion_id"))) else None
            t0, t1 = float(r["t_start"]), float(r["t_end"])

            baseline_mask, event_mask, b0 = build_event_and_baseline_masks(
                df=df,
                turns_df=turns_df,
                t0=t0,
                t1=t1,
                baseline_sec=BASELINE_WINDOW_SEC
            )

            mse_base = df.loc[baseline_mask, "mse"].to_numpy(dtype=float)
            mse_ev   = df.loc[event_mask, "mse"].to_numpy(dtype=float)

            ev_tag = f"evasion_{ev_id:02d}" if ev_id is not None else f"evasion_at_{t0:.2f}s"

            # CSV por evento: SOLO baseline local + evento
            out_ev_csv = os.path.join(OUT_DIR, f"{video_base}__{ev_tag}__mse.csv")
            df_event = df.loc[baseline_mask | event_mask, [
                "img_name","frame_idx","timestamp_sec","y_true_torque","y_pred_torque","mse","ae","re"
            ]].copy()

            df_event["segment"] = np.where(
                df_event["timestamp_sec"] < (t0 - EPS_TIME),
                f"baseline_{BASELINE_WINDOW_SEC:.0f}s",
                "esquiva_evento"
            )
            df_event.to_csv(out_ev_csv, index=False)

            # Histograma por evento

            # Extraer arrays de cada segmento desde el df del evento (robusto)
            base_df = df_event[df_event["segment"] == "baseline_3s"]
            event_df = df_event[df_event["segment"] == "esquiva_evento"]

            ae_base = base_df["ae"].to_numpy(dtype=float)
            ae_ev   = event_df["ae"].to_numpy(dtype=float)

            re_base = base_df["re"].to_numpy(dtype=float)
            re_ev   = event_df["re"].to_numpy(dtype=float)

            mse_base = base_df["mse"].to_numpy(dtype=float)
            mse_ev   = event_df["mse"].to_numpy(dtype=float)

            out_ae_png = os.path.join(OUT_DIR, f"{video_base}__{ev_tag}__hist_AE.png")
            plot_hist_metric(
                ae_base, ae_ev, out_ae_png,
                title=f"{video_base} | {ev_tag} | AE baseline({BASELINE_WINDOW_SEC:.0f}s) vs evento",
                metric_label="AE (|torque_pred - torque_true|)"
            )

            out_re_png = os.path.join(OUT_DIR, f"{video_base}__{ev_tag}__hist_RE.png")
            plot_hist_metric(
                re_base, re_ev, out_re_png,
                title=f"{video_base} | {ev_tag} | RE baseline({BASELINE_WINDOW_SEC:.0f}s) vs evento",
                metric_label=f"RE (AE / (|torque_true| + {EPS_RE}))"
            )
            out_mse_png = os.path.join(OUT_DIR, f"{video_base}__{ev_tag}__hist_MSE.png")
            plot_hist_metric(
                mse_base, mse_ev, out_mse_png,
                title=f"{video_base} | {ev_tag} | MSE baseline({BASELINE_WINDOW_SEC:.0f}s) vs evento",
                metric_label="MSE ((torque_pred - torque_true)^2)"
            )



            # Resumen
            base_stats = stats(mse_base)
            ev_stats   = stats(mse_ev)

            ae_base = df.loc[baseline_mask, "ae"].to_numpy(dtype=float)
            ae_ev   = df.loc[event_mask, "ae"].to_numpy(dtype=float)

            re_base = df.loc[baseline_mask, "re"].to_numpy(dtype=float)
            re_ev   = df.loc[event_mask, "re"].to_numpy(dtype=float)

            summary_rows.append({
                "video": video_base,
                "ev_tag": ev_tag,
                "t_start": t0,
                "t_end": t1,
                "duration_sec": float(r.get("duration_sec", np.nan)),
                "baseline_start": b0,
                "baseline_end": t0,
                "baseline_n": base_stats["n"],
                "baseline_mean": base_stats["mean"],
                "baseline_median": base_stats["median"],
                "baseline_p95": base_stats["p95"],
                "event_n": ev_stats["n"],
                "event_mean": ev_stats["mean"],
                "event_median": ev_stats["median"],
                "event_p95": ev_stats["p95"],
                "csv_event": out_ev_csv,
                "png_ae": out_ae_png,
                "png_re": out_re_png,
                "png_mse": out_mse_png,
                "baseline_ae_mean": np.mean(ae_base),
                "event_ae_mean": np.mean(ae_ev),
                "baseline_re_mean": np.mean(re_base),
                "event_re_mean": np.mean(re_ev),
            })

            print(f"  [EV] {ev_tag}: baseline_frames={base_stats['n']} | event_frames={ev_stats['n']} | p95_event={ev_stats['p95']:.2f}")

        # guardar summary del vídeo
        out_summary = os.path.join(OUT_DIR, f"{video_base}__summary_evasions.csv")
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
        print(f"  [OK] Summary: {out_summary}")

        
if __name__ == "__main__":
    main()

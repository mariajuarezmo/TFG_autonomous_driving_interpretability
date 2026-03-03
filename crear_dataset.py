import os
import re
import shutil
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN — ajusta estas rutas a tu caso
# ============================================================
RUN_NAME = "run4"  # Cambiar a "run1" o "run2" según la toma

pilotnet_datasets_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_datasets"
csvs_dir              = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales\csvs"
output_root           = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\Dataset"

FPS = 20  # FPS usado al extraer los frames (debe coincidir con sincronizar_videos_con_buscan.py)
TOLERANCE_SEC = 0.1  # tolerancia para merge_asof

# ============================================================
# RUTAS DE SALIDA
# ============================================================
run_dir          = os.path.join(output_root, RUN_NAME)
frames_out_dir   = os.path.join(run_dir, "frame_videos")
telemetry_out_dir = os.path.join(run_dir, "telemetry_csv")

os.makedirs(frames_out_dir, exist_ok=True)
os.makedirs(telemetry_out_dir, exist_ok=True)


# ============================================================
# PASO 1: Detectar y ordenar las carpetas de subvídeos
# ============================================================
def parse_folder(folder_name):
    """
    Extrae el prefijo base y el número de segmento de una carpeta.
    Ejemplo: '80f94eb526c7a9ac_00000012--e0ff8985d2--5--qcamera'
      -> prefijo = '80f94eb526c7a9ac_00000012--e0ff8985d2'
      -> segmento = 5
    """
    match = re.match(r'^(.+)--(\d+)--qcamera$', folder_name)
    if match:
        return match.group(1), int(match.group(2))
    return None, -1

# Recoger todas las carpetas válidas
all_subfolders = [
    f for f in os.listdir(pilotnet_datasets_dir)
    if os.path.isdir(os.path.join(pilotnet_datasets_dir, f))
    and parse_folder(f)[0] is not None
]

# Agrupar por prefijo, manteniendo el orden de primera aparición en el directorio
# y ordenando cada grupo internamente por número de segmento
from collections import OrderedDict
groups = OrderedDict()
for folder_name in all_subfolders:
    prefix, seg_num = parse_folder(folder_name)
    if prefix not in groups:
        groups[prefix] = []
    groups[prefix].append((seg_num, folder_name))

# Ordenar cada grupo por número de segmento
for prefix in groups:
    groups[prefix].sort(key=lambda x: x[0])

print(f"Grupos de toma detectados: {len(groups)}")
for prefix, segs in groups.items():
    print(f"  Prefijo: {prefix} — {len(segs)} segmentos (del {segs[0][0]} al {segs[-1][0]})")

# ---------------------------------------------------------------
# ASIGNACIÓN DE GRUPOS A RUNS
# Cada entrada indica qué índices de grupo (0-based, en orden de
# aparición en la carpeta) forman cada run. Si un run tiene varios
# grupos, se concatenan en el orden indicado.
# ---------------------------------------------------------------
RUNS_CONFIG = {
    "run4": [0],
 # ambos grupos pertenecen a run3
    # Ejemplos para otros casos:
    # "run1": [0],
    # "run2": [1],
    # "run3": [2],
}

if RUN_NAME not in RUNS_CONFIG:
    raise ValueError(f"RUN_NAME='{RUN_NAME}' no está en RUNS_CONFIG. Opciones: {list(RUNS_CONFIG.keys())}")

group_list = list(groups.items())
all_folders = []
for group_idx in RUNS_CONFIG[RUN_NAME]:
    prefix, segs = group_list[group_idx]
    print(f"\nGrupo {group_idx} → prefijo: {prefix} ({len(segs)} segmentos)")
    for seg_num, folder_name in segs:
        print(f"  [{seg_num}] {folder_name}")
        all_folders.append(folder_name)

print(f"\nTotal segmentos para '{RUN_NAME}': {len(all_folders)}")


# ============================================================
# PASO 2: Concatenar frames y data.txt unificado
# ============================================================
print("\n--- Copiando frames y construyendo data.txt unificado ---")

global_frame_idx = 0
data_txt_rows = []  # lista de (global_frame_id_str, torque)

for folder_name in all_folders:
    _, seg_num = parse_folder(folder_name)
    folder_path = os.path.join(pilotnet_datasets_dir, folder_name)
    data_txt_path = os.path.join(folder_path, "data.txt")

    # Leer el data.txt del segmento: {local_filename: torque}
    local_torque = {}
    if os.path.exists(data_txt_path):
        with open(data_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    local_torque[parts[0]] = parts[1]
    else:
        print(f"  AVISO: no se encontró data.txt en segmento {seg_num}")

    # Obtener frames del segmento ordenados
    jpg_files = sorted([
        fn for fn in os.listdir(folder_path)
        if fn.endswith(".jpg")
    ])

    print(f"  Segmento {seg_num}: {len(jpg_files)} frames, desde global {global_frame_idx:05d}")

    for local_filename in jpg_files:
        global_filename = f"{global_frame_idx:05d}.jpg"

        # Copiar frame con nuevo nombre global
        src = os.path.join(folder_path, local_filename)
        dst = os.path.join(frames_out_dir, global_filename)
        shutil.copy2(src, dst)

        # Añadir entrada al data.txt unificado si tiene torque
        torque = local_torque.get(local_filename)
        if torque is not None:
            data_txt_rows.append((global_filename, torque))

        global_frame_idx += 1

total_frames = global_frame_idx
print(f"\nTotal frames copiados: {total_frames}")

# Guardar data.txt unificado
data_txt_out = os.path.join(frames_out_dir, "data.txt")
with open(data_txt_out, "w") as f:
    for filename, torque in data_txt_rows:
        f.write(f"{filename} {torque}\n")
print(f"data.txt unificado guardado: {len(data_txt_rows)} entradas con torque de {total_frames} frames totales.")


# ============================================================
# PASO 3: Construir telemetry unificado (una fila por frame)
# ============================================================
print("\n--- Construyendo telemetr unificado ---")

carstate_fields = [
    'vEgo', 'gas', 'brake', 'steeringAngleDeg', 'steeringTorque',
    'aEgo', 'yawRate', 'gearShifter', 'steeringRateDeg', 'vEgoRaw',
    'standstill', 'leftBlinker', 'rightBlinker', 'gasPressed',
    'brakePressed', 'steeringPressed'
]

def find_carstate_csv(folder_name):
    """
    Busca el CSV de carState correspondiente a este segmento.
    folder_name ej: 80f94eb526c7a9ac_00000012--e0ff8985d2--5--qcamera
    CSV esperado:   80f94eb526c7a9ac_00000012--e0ff8985d2--5--qlog_carState.csv
    """
    # Quitar '--qcamera' del final para obtener la base del segmento
    base = re.sub(r'--qcamera$', '', folder_name)
    candidates = [
        base + "--qlog_carState.csv",
        base + "--rlog_carState.csv",
        base + "--qlog._carState.csv",
        base + "--rlog._carState.csv",
    ]
    for name in candidates:
        path = os.path.join(csvs_dir, name)
        if os.path.exists(path):
            return path
    return None

all_telemetry_rows = []
global_frame_idx = 0  # reiniciamos para recorrer en el mismo orden

for folder_name in all_folders:
    _, seg_num = parse_folder(folder_name)
    folder_path = os.path.join(pilotnet_datasets_dir, folder_name)

    jpg_files = sorted([fn for fn in os.listdir(folder_path) if fn.endswith(".jpg")])
    n_frames = len(jpg_files)

    # Buscar CSV de carState para este segmento
    carstate_csv_path = find_carstate_csv(folder_name)
    if carstate_csv_path is None:
        print(f"  AVISO: no se encontró carState CSV para segmento {seg_num}. Se rellenarán NaN.")
        # Crear filas vacías
        for i in range(n_frames):
            row = {"frame_id": f"{global_frame_idx:05d}.jpg"}
            for field in carstate_fields:
                row[field] = None
            all_telemetry_rows.append(row)
            global_frame_idx += 1
        continue

    print(f"  Segmento {seg_num}: usando {os.path.basename(carstate_csv_path)}")

    # Cargar carState CSV
    carstate = pd.read_csv(carstate_csv_path)

    # Calcular timestamp relativo en segundos desde el inicio del segmento
    if "t" not in carstate.columns:
        print(f"    AVISO: columna 't' no encontrada en CSV. Se rellenarán NaN.")
        for i in range(n_frames):
            row = {"frame_id": f"{global_frame_idx:05d}.jpg"}
            for field in carstate_fields:
                row[field] = None
            all_telemetry_rows.append(row)
            global_frame_idx += 1
        continue

    carstate["timestamp_sec"] = (carstate["t"] - carstate["t"].iloc[0]) / 1e9
    carstate = carstate.sort_values("timestamp_sec").reset_index(drop=True)

    # Crear DataFrame de frames con timestamp relativo según FPS
    frame_timestamps = [i / FPS for i in range(n_frames)]
    df_frames = pd.DataFrame({
        "timestamp_sec": frame_timestamps,
        "frame_local": jpg_files
    })

    # Sincronizar frames con carState usando merge_asof
    cols_to_merge = ["timestamp_sec"] + [c for c in carstate_fields if c in carstate.columns]
    df_sync = pd.merge_asof(
        df_frames.sort_values("timestamp_sec"),
        carstate[cols_to_merge],
        on="timestamp_sec",
        direction="nearest",
        tolerance=TOLERANCE_SEC
    )

    valid_pct = df_sync[carstate_fields[0]].notna().mean() * 100 if carstate_fields[0] in df_sync.columns else 0
    print(f"    {n_frames} frames, {valid_pct:.1f}% con datos de carState sincronizados.")

    # Construir filas del telemetry con frame_id global
    for _, row in df_sync.iterrows():
        tel_row = {"frame_id": f"{global_frame_idx:05d}.jpg"}
        for field in carstate_fields:
            tel_row[field] = row.get(field, None)
        all_telemetry_rows.append(tel_row)
        global_frame_idx += 1

# Guardar CSV unificado
df_telemetry = pd.DataFrame(all_telemetry_rows, columns=["frame_id"] + carstate_fields)
telemetry_csv_out = os.path.join(telemetry_out_dir, "telemetry.csv")
df_telemetry.to_csv(telemetry_csv_out, index=False)

print(f"\ntelemetry.csv guardado: {len(df_telemetry)} filas, {len(df_telemetry.columns)} columnas.")
print(f"Columnas: {list(df_telemetry.columns)}")
print(f"\nDataset '{RUN_NAME}' creado en: {run_dir}")
print(f"   frame_videos/  → {total_frames} frames + data.txt")
print(f"   telemetry_data/ → telemetry.csv")
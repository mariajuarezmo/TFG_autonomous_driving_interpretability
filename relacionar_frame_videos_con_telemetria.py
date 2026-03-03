"""
Script para enriquecer el Dataset con frames y telemetría por cada clip de vídeo.

Para cada runN dentro de Dataset/, y para cada clip .mp4 dentro de
evasion_driving_videos/ y normal_driving_videos/, genera una subcarpeta con:

  <clip_name>/
  ├── <clip_name>.mp4        ← el vídeo original (se mueve aquí)
  ├── <clip_name>.csv        ← filas del carState CSV correspondientes al clip
  └── frames/
      ├── 00000.jpg          ← frames extraídos del clip, renumerados desde 0
      ├── 00001.jpg
      └── ...

El t_start y t_end se extraen directamente del nombre del clip (últimos dos tokens).
La telemetría se obtiene del CSV individual de la carpeta csvs/ de cada run,
filtrando por el intervalo [t_start, t_end].

CONFIGURACIÓN: ajusta DATASET_DIR y CARSTATE_CSVS_POR_RUN.
"""

import os
import re
import shutil
import subprocess
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================
DATASET_DIR = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\Dataset"
FPS         = 20

# Ruta a la carpeta csvs/ de cada run.
# Ajusta cada ruta a donde guardaste los jsons_finales de cada toma.
CARSTATE_CSVS_POR_RUN = {
    "run1": r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales_1\csvs",
    "run2": r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales_2\csvs",
    "run3": r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales_3\csvs",
    "run4": r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales_4\csvs",
}

# Campos de telemetría a incluir en el CSV del clip
CARSTATE_FIELDS = [
    'vEgo', 'gas', 'brake', 'steeringAngleDeg', 'steeringTorque',
    'aEgo', 'yawRate', 'gearShifter', 'steeringRateDeg', 'vEgoRaw',
    'standstill', 'leftBlinker', 'rightBlinker', 'gasPressed',
    'brakePressed', 'steeringPressed'
]

# ============================================================
# UTILIDADES
# ============================================================

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg no está instalado o no está en el PATH.")
        return False


def parse_clip_name(clip_name):
    """
    Extrae video_id, t_start y t_end del nombre del clip.
    Los dos últimos tokens numéricos (separados por _) son siempre t_start y t_end.

    Ejemplos:
      80f94eb526c7a9ac_0000009a--9f194eb0e9--13_1_0.198_6.798.mp4
        -> video_id = 80f94eb526c7a9ac_0000009a--9f194eb0e9--13
           t_start  = 0.198,  t_end = 6.798

      80f94eb526c7a9ac_0000009a--9f194eb0e9--1_normal_1_0.0_47.4.mp4
        -> video_id = 80f94eb526c7a9ac_0000009a--9f194eb0e9--1
           t_start  = 0.0,    t_end = 47.4
    """
    stem  = clip_name.replace(".mp4", "")
    parts = stem.split("_")
    try:
        t_end   = float(parts[-1])
        t_start = float(parts[-2])
    except (ValueError, IndexError):
        return None, None, None

    # El video_id siempre termina en el token con "--<número>" (segmento del subvídeo)
    match = re.match(r'^(.+--\d+)', stem)
    if match:
        video_id = match.group(1)
    else:
        video_id = "_".join(parts[:-3])

    return video_id, t_start, t_end


def find_carstate_csv(video_id, csvs_dir):
    """
    Busca el CSV de carState para este video_id en csvs_dir.
    """
    candidates = [
        video_id + "--qlog_carState.csv",
        video_id + "--rlog_carState.csv",
        video_id + "--qlog._carState.csv",
        video_id + "--rlog._carState.csv",
    ]
    for name in candidates:
        path = os.path.join(csvs_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_carstate(csv_path):
    """
    Carga el CSV de carState y calcula timestamp_sec relativo al inicio del subvídeo.
    """
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError(f"El CSV {csv_path} no tiene columna 't'")
    df["timestamp_sec"] = (df["t"] - df["t"].iloc[0]) / 1e9
    return df.sort_values("timestamp_sec").reset_index(drop=True)


def extract_telemetry_for_clip(carstate_df, t_start, t_end):
    """
    Extrae y sincroniza las filas de telemetría para el intervalo [t_start, t_end].
    Usa merge_asof para asignar a cada frame (a 20 FPS) el valor de telemetría
    más cercano en el tiempo.
    Devuelve un DataFrame con frame_id (local, desde 00000.jpg) + campos de telemetría.
    """
    duration  = t_end - t_start
    n_frames  = max(1, round(duration * FPS))
    frame_ts  = [t_start + i / FPS for i in range(n_frames)]
    df_frames = pd.DataFrame({"timestamp_sec": frame_ts})

    margin   = 1.0
    cs_clip  = carstate_df[
        (carstate_df["timestamp_sec"] >= t_start - margin) &
        (carstate_df["timestamp_sec"] <= t_end   + margin)
    ].copy()

    cols    = ["timestamp_sec"] + [f for f in CARSTATE_FIELDS if f in cs_clip.columns]
    df_sync = pd.merge_asof(
        df_frames.sort_values("timestamp_sec"),
        cs_clip[cols],
        on="timestamp_sec",
        direction="nearest",
        tolerance=0.5
    )

    df_sync.insert(0, "frame_id", [f"{i:05d}.jpg" for i in range(len(df_sync))])
    df_sync = df_sync.drop(columns=["timestamp_sec"])
    return df_sync


def extract_frames_from_clip(clip_path, frames_dir):
    """Extrae todos los frames del clip como JPEGs renumerados desde 00000."""
    os.makedirs(frames_dir, exist_ok=True)
    out_pattern = os.path.join(frames_dir, "%05d.jpg")
    cmd = [
        "ffmpeg", "-i", clip_path,
        "-start_number", "0",
        "-q:v", "2",
        "-y", out_pattern
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"        ERROR ffmpeg: {e.stderr.decode()[:200]}")
        return False


def process_clip(clip_name, clips_dir, csvs_dir):
    """
    Procesa un clip individual:
      1. Parsea video_id, t_start, t_end del nombre
      2. Busca y carga el carState CSV del subvídeo
      3. Mueve el .mp4 a su subcarpeta
      4. Genera el CSV de telemetría sincronizado
      5. Extrae los frames
    Devuelve 'ok', 'skip' o 'error'.
    """
    clip_stem   = clip_name.replace(".mp4", "")
    clip_folder = os.path.join(clips_dir, clip_stem)
    new_mp4     = os.path.join(clip_folder, clip_name)

    # Comprobar si ya está procesado completamente
    if os.path.isdir(clip_folder) and os.path.exists(new_mp4):
        csv_path   = os.path.join(clip_folder, f"{clip_stem}.csv")
        frames_dir = os.path.join(clip_folder, "frames")
        if (os.path.exists(csv_path) and
                os.path.exists(frames_dir) and
                len(os.listdir(frames_dir)) > 0):
            print(f"      Ya procesado, omitiendo: {clip_name}")
            return "skip"

    # Parsear nombre
    video_id, t_start, t_end = parse_clip_name(clip_name)
    if video_id is None:
        print(f"      ERROR: no se pudo parsear '{clip_name}'")
        return "error"

    print(f"      {clip_name}")
    print(f"        video_id : {video_id}")
    print(f"        intervalo: {t_start}s -> {t_end}s  ({round(t_end - t_start, 2)}s)")

    # Buscar CSV de carState en la carpeta csvs del run correspondiente
    carstate_csv_path = find_carstate_csv(video_id, csvs_dir)
    if carstate_csv_path is None:
        print(f"        DESCARTADO: no hay CSV de telemetría para '{video_id}', borrando clip.")
        clip_path = os.path.join(clips_dir, clip_name)
        if os.path.exists(clip_path):
            os.remove(clip_path)
            print(f"        ✓ Clip borrado: {clip_name}")
        return "discarded"

    try:
        carstate_df = load_carstate(carstate_csv_path)
    except Exception as e:
        print(f"        ERROR cargando carState CSV: {e}")
        return "error"

    # Crear subcarpeta y mover .mp4
    os.makedirs(clip_folder, exist_ok=True)
    clip_path = os.path.join(clips_dir, clip_name)
    if not os.path.exists(new_mp4):
        shutil.move(clip_path, new_mp4)
    clip_path = new_mp4

    # Generar CSV de telemetría
    csv_path = os.path.join(clip_folder, f"{clip_stem}.csv")
    if not os.path.exists(csv_path):
        clip_tel = extract_telemetry_for_clip(carstate_df, t_start, t_end)
        clip_tel.to_csv(csv_path, index=False)
        print(f"        ✓ CSV: {len(clip_tel)} filas")
    else:
        print(f"        CSV ya existe")

    # Extraer frames
    frames_dir = os.path.join(clip_folder, "frames")
    if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
        if extract_frames_from_clip(clip_path, frames_dir):
            n = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            print(f"        ✓ Frames: {n} extraídos en frames/")
        else:
            return "error"
    else:
        print(f"        Frames ya extraídos")

    return "ok"


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def process_dataset():
    if not check_ffmpeg():
        return

    run_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if re.match(r"run\d+$", d) and os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not run_dirs:
        print(f"No se encontraron carpetas runN en {DATASET_DIR}")
        return

    print(f"Runs encontrados: {run_dirs}\n")

    for run_name in run_dirs:
        run_path     = os.path.join(DATASET_DIR, run_name)
        evasions_dir = os.path.join(run_path, "video_data", "evasion_driving_videos")
        normal_dir   = os.path.join(run_path, "video_data", "normal_driving_videos")

        # Obtener la carpeta de CSVs para este run
        csvs_dir = CARSTATE_CSVS_POR_RUN.get(run_name)
        if csvs_dir is None:
            print(f"  AVISO: no hay ruta de CSVs configurada para {run_name}, saltando.")
            continue
        if not os.path.exists(csvs_dir):
            print(f"  ERROR: la carpeta de CSVs no existe para {run_name}: {csvs_dir}")
            continue

        print(f"{'='*60}")
        print(f"Procesando {run_name}...")
        print(f"  CSVs: {csvs_dir}")

        counters = {"ok": 0, "skip": 0, "error": 0, "discarded": 0}

        for clip_type, clips_dir in [("ESQUIVA", evasions_dir), ("NORMAL", normal_dir)]:
            if not os.path.exists(clips_dir):
                print(f"  Carpeta no encontrada: {clips_dir}")
                continue

            mp4_files = sorted([
                f for f in os.listdir(clips_dir)
                if f.endswith(".mp4") and os.path.isfile(os.path.join(clips_dir, f))
            ])

            print(f"\n  [{clip_type}] {len(mp4_files)} clips en {os.path.basename(clips_dir)}/")

            for clip_name in mp4_files:
                result = process_clip(clip_name, clips_dir, csvs_dir)
                counters[result] += 1

        print(f"\n  Resumen {run_name}:")
        print(f"    Procesados:  {counters['ok']}")
        print(f"    Omitidos:    {counters['skip']}")
        print(f"    Descartados: {counters['discarded']}")
        print(f"    Errores:     {counters['error']}")

    print(f"\n{'='*60}")
    print("Proceso completado!")
    print(f"\nEstructura generada por clip:")
    print("  <clip_name>/")
    print("  ├── <clip_name>.mp4   (video)")
    print("  ├── <clip_name>.csv   (telemetria sincronizada, frame_id desde 00000.jpg)")
    print("  └── frames/")
    print("      ├── 00000.jpg")
    print("      └── ...")


if __name__ == "__main__":
    print("="*60)
    print("ENRIQUECEDOR DE CLIPS  (frames + telemetria por clip)")
    print("="*60)
    print()
    process_dataset()
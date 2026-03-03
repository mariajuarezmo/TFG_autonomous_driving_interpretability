"""
Script para extraer fragmentos de video basados en datos de análisis de giros.

Para cada video original extrae:
  - Un fragmento por cada esquiva detectada en _ALL_TURNS.csv
  - Uno o más fragmentos de conducción normal (los intervalos del video
    que NO están cubiertos por ninguna esquiva)

Los fragmentos de esquiva se guardan en:   OUTPUT_DIR/evasions/
Los fragmentos de conducción normal en:    OUTPUT_DIR/normal/
"""

import pandas as pd
import os
import subprocess

# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================
BASE_DIR    = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo"
CSV_FILE    = os.path.join(BASE_DIR, r"jsons_finales\csvs\_turn_analysis_out\_ALL_TURNS.csv")
VIDEOS_DIR  = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR  = os.path.join(BASE_DIR, r"jsons_finales\csvs\_turn_analysis_out")

# Duración mínima (segundos) de un fragmento de conducción normal para guardarlo.
# Fragmentos más cortos se descartan (evita generar clips de 0.2s entre esquivas seguidas).
MIN_NORMAL_DURATION_SEC = 2.0

# ============================================================
# UTILIDADES
# ============================================================

def check_ffmpeg():
    """Verifica si ffmpeg está instalado."""
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg no está instalado o no está en el PATH.")
        print("Instálalo desde: https://ffmpeg.org/download.html")
        return False


def get_video_duration(video_path):
    """
    Devuelve la duración del vídeo en segundos usando ffprobe.
    Devuelve None si no se puede obtener.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return float(result.stdout.decode().strip())
    except Exception:
        return None


def extract_video_chunk(input_video, output_video, t_start, t_end):
    """
    Extrae un fragmento de vídeo usando ffmpeg.

    Args:
        input_video:  Ruta al vídeo original.
        output_video: Ruta de salida.
        t_start:      Tiempo de inicio en segundos.
        t_end:        Tiempo de fin en segundos.

    Returns:
        True si la extracción fue exitosa, False en caso contrario.
    """
    duration = round(t_end - t_start, 3)
    if duration <= 0:
        return False

    cmd = [
        "ffmpeg",
        "-ss", str(t_start),
        "-i", input_video,
        "-t", str(duration),
        "-c", "copy",
        "-y",
        output_video
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR ffmpeg: {e.stderr.decode()[:200]}")
        return False


def normal_intervals(evasion_intervals, video_duration):
    """
    Dado un conjunto de intervalos de esquiva (lista de (t_start, t_end))
    y la duración total del vídeo, devuelve los intervalos complementarios
    (conducción normal).

    Args:
        evasion_intervals: Lista de tuplas (t_start, t_end), ordenadas por t_start.
        video_duration:    Duración total del vídeo en segundos.

    Returns:
        Lista de tuplas (t_start, t_end) de conducción normal.
    """
    intervals = sorted(evasion_intervals, key=lambda x: x[0])

    normal = []
    cursor = 0.0

    for t_start, t_end in intervals:
        if cursor < t_start:
            normal.append((round(cursor, 3), round(t_start, 3)))
        cursor = max(cursor, t_end)

    # Tramo final tras la última esquiva
    if cursor < video_duration:
        normal.append((round(cursor, 3), round(video_duration, 3)))

    # Filtrar fragmentos demasiado cortos
    normal = [(s, e) for s, e in normal if (e - s) >= MIN_NORMAL_DURATION_SEC]

    return normal


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def process_turns_csv():

    if not check_ffmpeg():
        return

    # Verificar existencia de archivos y carpetas
    for path, label in [(CSV_FILE, "CSV"), (VIDEOS_DIR, "directorio de vídeos")]:
        if not os.path.exists(path):
            print(f"ERROR: No se encontró el {label}: {path}")
            return

    # Crear subcarpetas de salida
    evasions_dir = os.path.join(OUTPUT_DIR, "evasions")
    normal_dir   = os.path.join(OUTPUT_DIR, "normal")
    os.makedirs(evasions_dir, exist_ok=True)
    os.makedirs(normal_dir,   exist_ok=True)

    # Leer CSV de esquivas
    print(f"Leyendo CSV: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"ERROR al leer el CSV: {e}")
        return

    print(f"Total de registros en el CSV: {len(df)}")
    print(f"Columnas: {list(df.columns)}\n")

    # Normalizar video_id (quitar sufijos de CSV si los hubiera)
    def clean_video_id(raw):
        vid = str(raw)
        for suffix in ["--rlog._carState.csv", "--rlog_carState.csv",
                       "--qlog._carState.csv", "--qlog_carState.csv"]:
            if vid.endswith(suffix):
                vid = vid[:-len(suffix)]
                break
        if vid.endswith(".csv"):
            vid = os.path.splitext(vid)[0]
        return vid

    df["video_id_clean"] = df["video"].apply(clean_video_id)

    # Agrupar por vídeo para poder calcular los intervalos normales
    grouped = df.groupby("video_id_clean")

    counters = {"evasion_ok": 0, "evasion_skip": 0, "evasion_err": 0,
                "normal_ok": 0,  "normal_skip": 0,  "normal_err": 0}

    all_video_ids = sorted(df["video_id_clean"].unique())

    for video_id in all_video_ids:
        video_filename   = f"{video_id}--qcamera.ts"
        input_video_path = os.path.join(VIDEOS_DIR, video_filename)

        if not os.path.exists(input_video_path):
            print(f"ADVERTENCIA: vídeo no encontrado → {video_filename}")
            continue

        # Obtener duración real del vídeo
        video_duration = get_video_duration(input_video_path)
        if video_duration is None:
            print(f"ADVERTENCIA: no se pudo obtener duración de {video_filename}, se usará t_end máximo.")

        rows = grouped.get_group(video_id)
        evasion_intervals = []

        print(f"\n{'='*60}")
        print(f"Vídeo: {video_id}")
        print(f"  Duración: {video_duration:.2f}s" if video_duration else "  Duración: desconocida")
        print(f"  Esquivas detectadas: {len(rows)}")

        # ── 1. Fragmentos de ESQUIVA ──────────────────────────────
        for _, row in rows.iterrows():
            t_start    = round(float(row["t_start"]), 3)
            t_end      = round(float(row["t_end"]),   3)
            evasion_id = row["evasion_id"]

            evasion_intervals.append((t_start, t_end))

            out_name  = f"{video_id}_{evasion_id}_{t_start}_{t_end}.mp4"
            out_path  = os.path.join(evasions_dir, out_name)

            if os.path.exists(out_path):
                print(f"  [ESQUIVA] Ya existe, omitiendo: {out_name}")
                counters["evasion_skip"] += 1
                continue

            print(f"  [ESQUIVA] {t_start}s → {t_end}s  ({round(t_end-t_start,2)}s)")
            if extract_video_chunk(input_video_path, out_path, t_start, t_end):
                print(f"    ✓ Guardado: {out_name}")
                counters["evasion_ok"] += 1
            else:
                print(f"    ✗ Error al extraer")
                counters["evasion_err"] += 1

        # ── 2. Fragmentos de CONDUCCIÓN NORMAL ───────────────────
        # Si no tenemos duración real, usamos el t_end máximo del CSV como aproximación
        if video_duration is None:
            video_duration = round(float(rows["t_end"].max()) + 1.0, 3)

        normal_segs = normal_intervals(evasion_intervals, video_duration)
        print(f"  Segmentos de conducción normal: {len(normal_segs)}")

        for seg_idx, (ns, ne) in enumerate(normal_segs, start=1):
            out_name = f"{video_id}_normal_{seg_idx}_{ns}_{ne}.mp4"
            out_path = os.path.join(normal_dir, out_name)

            if os.path.exists(out_path):
                print(f"  [NORMAL]  Ya existe, omitiendo: {out_name}")
                counters["normal_skip"] += 1
                continue

            print(f"  [NORMAL]  {ns}s → {ne}s  ({round(ne-ns,2)}s)")
            if extract_video_chunk(input_video_path, out_path, ns, ne):
                print(f"    ✓ Guardado: {out_name}")
                counters["normal_ok"] += 1
            else:
                print(f"    ✗ Error al extraer")
                counters["normal_err"] += 1

    # ── Vídeos sin esquivas (no aparecen en el CSV) ───────────────
    # Para estos, todo el vídeo es conducción normal
    print(f"\n{'='*60}")
    print("Buscando vídeos sin ninguna esquiva detectada...")
    all_ts_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith("--qcamera.ts")]

    for ts_file in sorted(all_ts_files):
        video_id = ts_file.replace("--qcamera.ts", "")
        if video_id in all_video_ids:
            continue  # ya procesado

        input_video_path = os.path.join(VIDEOS_DIR, ts_file)
        video_duration   = get_video_duration(input_video_path)

        if video_duration is None:
            print(f"  ADVERTENCIA: no se pudo obtener duración de {ts_file}")
            continue

        if video_duration < MIN_NORMAL_DURATION_SEC:
            continue

        out_name = f"{video_id}_normal_1_0.0_{round(video_duration,3)}.mp4"
        out_path = os.path.join(normal_dir, out_name)

        if os.path.exists(out_path):
            print(f"  [NORMAL]  Ya existe, omitiendo: {out_name}")
            counters["normal_skip"] += 1
            continue

        print(f"  [NORMAL SIN ESQUIVA] {video_id}  (0s → {video_duration:.2f}s)")
        if extract_video_chunk(input_video_path, out_path, 0.0, video_duration):
            print(f"    ✓ Guardado: {out_name}")
            counters["normal_ok"] += 1
        else:
            print(f"    ✗ Error al extraer")
            counters["normal_err"] += 1

    # ── Resumen ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Fragmentos de ESQUIVA   → OK: {counters['evasion_ok']}  |  Error: {counters['evasion_err']}  |  Omitidos: {counters['evasion_skip']}")
    print(f"  Fragmentos de NORMAL    → OK: {counters['normal_ok']}  |  Error: {counters['normal_err']}  |  Omitidos: {counters['normal_skip']}")
    print(f"\n  Salida esquivas:  {evasions_dir}")
    print(f"  Salida normales:  {normal_dir}")


if __name__ == "__main__":
    print("="*60)
    print("EXTRACTOR DE FRAGMENTOS DE VÍDEO  (esquivas + conducción normal)")
    print("="*60)
    print()
    process_turns_csv()
    print("\n¡Proceso completado!")
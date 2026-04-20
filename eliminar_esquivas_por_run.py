import os
import pandas as pd
import shutil
from pathlib import Path

def filtrar_esquivas_por_run(dataset_path, output_path, run_target):
    """
    Crea una copia de un run específico eliminando los frames de esquiva.
    """
    run_path = Path(dataset_path) / run_target
    new_run_path = Path(output_path) / f"{run_target}_sin_esquivas"
    
    print(f"--- Procesando {run_target} ---")
    
    # 1. Identificar frames de esquiva (Lista Negra)
    # Buscamos en todas las subcarpetas de evasion_driving_videos
    evasion_dir = run_path / "video_data" / "evasion_driving_videos"
    frames_a_borrar = set()
    
    if evasion_dir.exists():
        for clip_folder in evasion_dir.iterdir():
            if clip_folder.is_dir():
                # El README dice que frame_id en el CSV coincide con los archivos
                csv_path = clip_folder / f"{clip_folder.name}.csv"
                if csv_path.exists():
                    df_evasion = pd.read_csv(csv_path)
                    frames_a_borrar.update(df_evasion['frame_id'].astype(str).tolist())

    print(f"Frames de esquiva identificados en {run_target}: {len(frames_a_borrar)}")

    # 2. Crear estructura de carpetas de salida
    (new_run_path / "telemetry_data").mkdir(parents=True, exist_ok=True)
    (new_run_path / "video_data" / "frame_videos").mkdir(parents=True, exist_ok=True)

    # 3. Filtrar y copiar telemetry.csv
    telemetry_old = run_path / "telemetry_data" / "telemetry.csv"
    if telemetry_old.exists():
        df_telemetry = pd.read_csv(telemetry_old)
        # Solo nos quedamos con lo que NO está en la lista negra [cite: 1886]
        df_filtrado = df_telemetry[~df_telemetry['frame_id'].isin(frames_a_borrar)]
        df_filtrado.to_csv(new_run_path / "telemetry_data" / "telemetry.csv", index=False)
        print(f"Telemetría filtrada: de {len(df_telemetry)} a {len(df_filtrado)} filas.")

    # 4. Copiar solo frames de "Normal Driving"
    # Usamos frame_videos como fuente y filtramos
    src_frames = run_path / "video_data" / "frame_videos"
    dest_frames = new_run_path / "video_data" / "frame_videos"
    
    count_frames = 0
    for img in src_frames.glob("*.jpg"):
        if img.name not in frames_a_borrar:
            shutil.copy2(img, dest_frames / img.name)
            count_frames += 1
            
    print(f"Frames copiados a {dest_frames}: {count_frames}")
    print(f"Completado: {new_run_path}\n")

# --- CONFIGURACIÓN ---
DATASET_ORIGINAL = "Dataset"  # Tu carpeta original
DATASET_EXPERIMENTO = "Dataset_Filtrado" # Carpeta donde se guardarán los resultados

# Ejecutar para cada run de forma independiente
for r in ["run1", "run2", "run3", "run4"]:
    filtrar_esquivas_por_run(DATASET_ORIGINAL, DATASET_EXPERIMENTO, r)
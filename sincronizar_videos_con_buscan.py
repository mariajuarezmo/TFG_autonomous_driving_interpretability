import os
import cv2
import pandas as pd
import numpy as np

# CONFIGURACIÓN 
videos_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\videos"
csvs_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales\csvs"
output_root = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_datasets"
fps_default = 20
tolerance_sec = 0.1  # tolerancia para merge_asof (medio segundo)

os.makedirs(output_root, exist_ok=True)

# Función principal 
def procesar_video(video_path, carstate_csv):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Procesando {base_name} ...")

    #  1. Cargar datos del Bus-CAN 
    carstate = pd.read_csv(carstate_csv)
    if "t" not in carstate.columns or "steeringTorque" not in carstate.columns:
        print(f"CSV {carstate_csv} no tiene las columnas esperadas.")
        return
    carstate["timestamp_sec"] = (carstate["t"] - carstate["t"].iloc[0]) / 1e9
    carstate = carstate.sort_values("timestamp_sec")

    # 2. Extraer frames del vídeo 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if np.isnan(fps) or fps <= 0:
        fps = fps_default
        print(f"  FPS no detectado, usando valor por defecto: {fps_default}")
    else:
        print(f"  FPS detectado: {fps:.2f}")

    out_dir = os.path.join(output_root, base_name)
    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    video_timestamps = []
    filenames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        filename = f"{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, filename), frame)
        video_timestamps.append(ts)
        filenames.append(filename)
        frame_idx += 1

    cap.release()
    print(f"  → {frame_idx} frames extraídos.")

    df_video = pd.DataFrame({"timestamp_sec": video_timestamps, "filename": filenames})

    # 3. Sincronización (merge_asof) 
    df_sync = pd.merge_asof(
        df_video.sort_values("timestamp_sec"),
        carstate[["timestamp_sec", "steeringTorque"]],
        on="timestamp_sec",
        direction="nearest",
        tolerance=tolerance_sec
    )

    porcentaje_valido = df_sync["steeringTorque"].notna().mean() * 100
    if porcentaje_valido < 50:
        print(f"  ADVERTENCIA: solo el {porcentaje_valido:.1f}% de frames tienen torque.")
        print(f"  Verifica que el vídeo y el CSV corresponden al mismo momento.")

    # 4. Guardar data.txt 
    data_txt_path = os.path.join(out_dir, "data.txt")
    with open(data_txt_path, "w") as f:
        for _, row in df_sync.iterrows():
            if pd.notna(row["steeringTorque"]):
                f.write(f"{row['filename']} {row['steeringTorque']}\n")

    frames_con_torque = df_sync["steeringTorque"].notna().sum()
    frames_sin_torque = df_sync["steeringTorque"].isna().sum()
    print(f"  → Frames con torque: {frames_con_torque}")
    print(f"  → Frames descartados (sin torque): {frames_sin_torque}")
    print(f"Dataset listo: {out_dir}\n")


# 5. Bucle sobre los vídeos
for file_name in os.listdir(videos_dir): 
    if file_name.endswith(".ts"): 
        
        video_path = os.path.join(videos_dir, file_name)

        if '--qcamera.ts' in file_name:
            
            # Base común (sin el sufijo del vídeo)
            base = file_name.replace("--qcamera.ts", "")

            # Posibles nombres de CSV (soporta ambos formatos)
            posibles_csv = [
                base + "--rlog_carState.csv",
                base + "--rlog._carState.csv",
                base + "--qlog_carState.csv",
                base + "--qlog._carState.csv"
            ]

            carstate_csv = None

            # Buscar cuál existe realmente
            for nombre_csv in posibles_csv:
                ruta_csv = os.path.join(csvs_dir, nombre_csv)
                if os.path.exists(ruta_csv):
                    carstate_csv = ruta_csv
                    break

            if carstate_csv is None:
                print(f"No se encontró CSV para {file_name}")
                continue

            # Procesar el vídeo
            procesar_video(video_path, carstate_csv)

print("Sincronización completada para todos los vídeos.")



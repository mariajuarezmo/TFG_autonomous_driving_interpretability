import os
import pandas as pd
from pathlib import Path

def generar_txt_filtrado(ruta_dataset_filtrado):
    base_path = Path(ruta_dataset_filtrado)
    
    # Recorremos cada carpeta runN_sin_esquivas
    for run_folder in base_path.iterdir():
        if run_folder.is_dir() and "_sin_esquivas" in run_folder.name:
            print(f"Generando .txt para: {run_folder.name}")
            
            csv_path = run_folder / "telemetry_data" / "telemetry.csv"
            txt_output = run_folder / "telemetry_data" / "frame-torque.txt"
            
            if csv_path.exists():
                # Leemos el CSV que ya filtramos antes
                df = pd.read_csv(csv_path, low_memory=False)
                
                # Creamos el contenido del TXT: <frame_id> <steeringTorque>
                # El script de entrenamiento espera exactamente este formato
                with open(txt_output, "w") as f:
                    for _, row in df.iterrows():
                        f.write(f"{row['frame_id']} {row['steeringTorque']}\n")
                
                print(f"  [OK] Creado: {txt_output} con {len(df)} líneas.")
            else:
                print(f"  [Error] No se encontró telemetry.csv en {run_folder}")

# Ejecútalo apuntando a tu carpeta de resultados
generar_txt_filtrado("Dataset_Filtrado")
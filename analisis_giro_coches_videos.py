"""
Script para extraer fragmentos de video basados en datos de análisis de giros.

Este script lee un CSV con información de giros detectados en videos
y extrae los fragmentos correspondientes de los videos originales.
"""

import pandas as pd
import os
from pathlib import Path
import subprocess
import sys

# Configuración de rutas
BASE_DIR = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo"
CSV_FILE = os.path.join(BASE_DIR, r"jsons_finales\csvs\_turn_analysis_out\_ALL_TURNS.csv")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, r"jsons_finales\csvs\_turn_analysis_out")

def check_ffmpeg():
    """Verifica si ffmpeg está instalado."""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: ffmpeg no está instalado o no está en el PATH.")
        print("Por favor, instala ffmpeg desde: https://ffmpeg.org/download.html")
        return False

def extract_video_chunk(input_video, output_video, t_start, t_end):
    """
    Extrae un fragmento de video usando ffmpeg.
    
    Args:
        input_video: Ruta al video original
        output_video: Ruta donde guardar el fragmento
        t_start: Tiempo de inicio en segundos
        t_end: Tiempo de fin en segundos
    
    Returns:
        True si la extracción fue exitosa, False en caso contrario
    """
    duration = (t_end - t_start) + 1
    
    # Comando ffmpeg para extraer el fragmento
    # -ss: posición de inicio
    # -t: duración
    # -c copy: copia los streams sin re-encodificar (más rápido)
    # Si hay problemas con -c copy, usa -c:v libx264 -c:a aac
    cmd = [
        "ffmpeg",
        "-ss", str(t_start),
        "-i", input_video,
        "-t", str(duration),
        "-c", "copy",
        "-y",  # Sobrescribir archivo de salida si existe
        output_video
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR al procesar: {e.stderr.decode()}")
        return False

def process_turns_csv():
    """Procesa el CSV y extrae todos los fragmentos de video."""
    
    # Verificar que ffmpeg esté disponible
    if not check_ffmpeg():
        return
    
    # Verificar que existan los directorios
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: No se encontró el archivo CSV: {CSV_FILE}")
        return
    
    if not os.path.exists(VIDEOS_DIR):
        print(f"ERROR: No se encontró el directorio de videos: {VIDEOS_DIR}")
        return
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creando directorio de salida: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    
    # Leer el CSV
    print(f"Leyendo CSV: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"ERROR al leer el CSV: {e}")
        return
    
    print(f"Total de registros en el CSV: {len(df)}")
    print(f"\nColumnas encontradas: {list(df.columns)}")
    print(f"\nPrimeras filas:")
    print(df.head())
    
    # Contadores
    successful = 0
    failed = 0
    skipped = 0
    
    # Procesar cada fila del CSV
    for idx, row in df.iterrows():
        video_id = str(row['video'])

        # Si viene con sufijos tipo "--rlog._carState.csv" o "--rlog_carState.csv", los quitamos
        for suffix in ["--rlog._carState.csv", "--rlog_carState.csv", "--qlog._carState.csv", "--qlog_carState.csv"]:
            if video_id.endswith(suffix):
                video_id = video_id[:-len(suffix)]
                break

        # Si por cualquier razón viene con ".csv" al final, lo quitamos también
        if video_id.endswith(".csv"):
            video_id = os.path.splitext(video_id)[0]

        evasion_id = row['evasion_id']
        t_start = round(row['t_start'], 3)  # Redondear a 3 decimales
        t_end = round(row['t_end'], 3)
        
        # Construir nombre del video original
        video_filename = f"{video_id}--qcamera.ts"
        input_video_path = os.path.join(VIDEOS_DIR, video_filename)
        
        # Construir nombre del archivo de salida
        # Formato: {video_id}_{evasion_id}_{t_start}_{t_end}.mp4
        output_filename = f"{video_id}_{evasion_id}_{t_start}_{t_end}.mp4"
        output_video_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"\n[{idx+1}/{len(df)}] Procesando: {video_id} (evasion {evasion_id})")
        print(f"  Tiempo: {t_start}s -> {t_end}s (duración: {round(t_end - t_start, 2)}s)")
        
        # Verificar que existe el video original
        if not os.path.exists(input_video_path):
            print(f"  ADVERTENCIA: No se encontró el video: {input_video_path}")
            skipped += 1
            continue
        
        # Verificar si el archivo de salida ya existe
        if os.path.exists(output_video_path):
            print(f"  INFO: El fragmento ya existe, omitiendo: {output_filename}")
            skipped += 1
            continue
        
        # Extraer el fragmento
        print(f"  Extrayendo fragmento...")
        if extract_video_chunk(input_video_path, output_video_path, t_start, t_end):
            print(f"  ✓ Fragmento guardado: {output_filename}")
            successful += 1
        else:
            print(f"  ✗ Error al extraer fragmento")
            failed += 1
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE PROCESAMIENTO")
    print("="*60)
    print(f"Total de registros procesados: {len(df)}")
    print(f"Fragmentos extraídos exitosamente: {successful}")
    print(f"Errores: {failed}")
    print(f"Omitidos (ya existían o video no encontrado): {skipped}")
    print(f"\nLos fragmentos se guardaron en: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("="*60)
    print("EXTRACTOR DE FRAGMENTOS DE VIDEO")
    print("="*60)
    print()
    
    process_turns_csv()
    
    print("\n¡Proceso completado!")
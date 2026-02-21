import os
import json
import csv

# --- Rutas de entrada y salida ---
input_folder = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales"
output_folder = os.path.join(input_folder, "csvs")
os.makedirs(output_folder, exist_ok=True)

# --- Definir columnas por tipo de mensaje ---

carState_fields = [
    't', 'vEgo', 'gas', 'brake', 'steeringAngleDeg', 'steeringTorque', 
    'aEgo', 'yawRate', 'gearShifter', 'steeringRateDeg', 'vEgoRaw',
    'standstill', 'leftBlinker', 'rightBlinker', 'gasPressed', 
    'brakePressed', 'steeringPressed'
]

def iter_entries(file_path):
    """
    Devuelve cada 'entry' del archivo soportando:
    - JSON completo: lista [...] o dict {...}
    - JSONL/NDJSON: un JSON por línea
    """
    # Intento 1: JSON completo
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            for e in obj:
                yield e
            return

        if isinstance(obj, dict):
            # Puede venir como un dict único o algo tipo {"entries":[...]}
            if "entries" in obj and isinstance(obj["entries"], list):
                for e in obj["entries"]:
                    yield e
            else:
                yield obj
            return

    except json.JSONDecodeError:
        pass  # No era JSON completo, probamos JSONL

    # Intento 2: JSONL (un JSON por línea)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue



def process_qlog_file(file_path):
    """Extrae carState en ambos formatos (qlog y rlog), sea .json o .jsonl."""
    carState_data = []

    for entry in iter_entries(file_path):
        if not isinstance(entry, dict):
            continue

        # Formato 1 (qlog): {"t":..., "type":"carState", "data":{...}}
        if entry.get("type") == "carState":
            t = entry.get("t")
            data = entry.get("data", {})

        # Formato 2 (rlog): {"which":"carState", "logMonoTime":..., "carState":{...}}
        elif entry.get("which") == "carState":
            t = entry.get("logMonoTime")  # equivalente a t
            data = entry.get("carState", {})
        
        elif "carState" in entry and "logMonoTime" in entry and isinstance(entry.get("carState"), dict):
            t = entry.get("logMonoTime")
            data = entry.get("carState", {})

        else:
            continue

        if not isinstance(data, dict):
            continue

        row = {
            't': t,
            'vEgo': data.get('vEgo'),
            'gas': data.get('gas'),
            'brake': data.get('brake'),
            'steeringAngleDeg': data.get('steeringAngleDeg'),
            'steeringTorque': data.get('steeringTorque'),
            'aEgo': data.get('aEgo'),
            'yawRate': data.get('yawRate'),
            'gearShifter': data.get('gearShifter'),
            'steeringRateDeg': data.get('steeringRateDeg'),
            'vEgoRaw': data.get('vEgoRaw'),
            'standstill': data.get('standstill'),
            'leftBlinker': data.get('leftBlinker'),
            'rightBlinker': data.get('rightBlinker'),
            'gasPressed': data.get('gasPressed'),
            'brakePressed': data.get('brakePressed'),
            'steeringPressed': data.get('steeringPressed')
        }
        carState_data.append(row)

    return carState_data


def save_to_csv(csv_path, data, fieldnames):
    """Guarda los datos en un CSV con las columnas especificadas."""
    if not data:
        print(f"No hay datos para {os.path.basename(csv_path)}")
        return
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"Generado: {os.path.basename(csv_path)} ({len(data)} filas)")


def analyze_missing_values(data, data_type):
    """Analiza si hay valores None dentro de los datos de un mismo tipo."""
    if not data:
        return
    
    missing_by_field = {}
    total_rows = len(data)
    
    for row in data:
        for key, value in row.items():
            if value is None:
                if key not in missing_by_field:
                    missing_by_field[key] = 0
                missing_by_field[key] += 1
    
    if missing_by_field:
        print(f"\n Valores None detectados en {data_type}:")
        for field, count in missing_by_field.items():
            percentage = (count / total_rows) * 100
            print(f"      {field}: {count}/{total_rows} ({percentage:.1f}%)")


# --- Procesar todos los archivos qlog.jsonl ---
print("Iniciando procesamiento de archivos qlog.jsonl...\n")

for file_name in os.listdir(input_folder):
    if (
        file_name.endswith('--qlog.json') or file_name.endswith('--rlog.json')
        or file_name.endswith('qlog.jsonl') or file_name.endswith('rlog.jsonl')
        or file_name.endswith('--qlog.jsonl') or file_name.endswith('--rlog.jsonl')
    ):
        file_path = os.path.join(input_folder, file_name)
        base_name = file_name
        if base_name.endswith('.jsonl'):
            base_name = base_name[:-5]
        elif base_name.endswith('.json'):
            base_name = base_name[:-5]

        print(f'Procesando: {file_name}')
        
        # Procesar el archivo
        carState_data= process_qlog_file(file_path)
        print(f"  -> Filas carState extraídas: {len(carState_data)}")

        
        # Analizar valores faltantes
        analyze_missing_values(carState_data, 'carState')
        
        # Guardar CSVs separados
        save_to_csv( 
            os.path.join(output_folder, f'{base_name}_carState.csv'),
            carState_data,
            carState_fields
        )
        print()

print("Todos los archivos procesados correctamente.")
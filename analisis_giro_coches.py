import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  RUTAS 
CARSTATE_DIR = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\jsons_finales\csvs"
OUT_DIR = os.path.join(CARSTATE_DIR, "_turn_analysis_out")

# PARÁMETROS BASE (histéresis)
'''
Evita detectar miles de "giros falsos" por pequeñas vibraciones. 
Es como un interruptor con dos niveles: uno para activar y otro para desactivar.
'''
ANGLE_ON_DEG = 10.0 # Si el volante supera 10°, empieza un "giro"
ANGLE_OFF_DEG = 6.0 # Si baja de 6°, termina el "giro"

# DURACIÓN TOTAL 
MIN_TOTAL_EVASION_SEC = 2.0 # Una esquiva dura mínimo 1 segundo
MAX_TOTAL_EVASION_SEC = 8.0 # Y máximo 8 segundos
ENFORCE_TOTAL_DURATION = True

# GAP máximo entre eventos si buscamos doble-evento
# Si el segundo giro empieza mucho más tarde, ya no lo consideras parte de la misma esquiva.
MAX_GAP_BETWEEN_PEAKS_SEC = 2.00

# FILTROS DE "PICO 1"
P1_MIN_SUBTURN_SEC = 0.15 # El primer giro debe durar al menos 0.15s
P1_MAX_SUBTURN_SEC = 4.50 # Y como mucho 4.5s
P1_PEAK_MIN_ABS_ANGLE_DEG = 40.0 # El primer giro debe superar 40°
P1_PEAK_MAX_ABS_ANGLE_DEG = 140.0 # Pero no más de 140°

#DETECTOR DE PICOS AISLADOS
# Rango del pico 
PEAK_MIN_ABS_DEG = 65.0 # El pico debe superar 65°  
PEAK_MAX_ABS_DEG = 180.0 # Pero no más de 180° (límite físico)

# Duración típica del pico 
PEAK_MIN_DUR_SEC = 1.5  # Debe durar al menos 1.5s
PEAK_MAX_DUR_SEC = 5.0  # Debe durar como mucho 5s

# Nivel base para aislamiento
BASELINE_MAX_DEG = 65.0  # El ángulo máximo que se considera "conducción normal" (sin maniobra especial).   
'''
Ángulo (°)
    100 |        /\         ← PICO (esquiva)
     80 |       /  \
     65 | ─────/────\───── ← BASELINE (línea de corte)
     40 |    /        \
     20 |   /          \___
      0 |__/________________
        Tiempo (s)
'''
'''
Antes y después del pico, el ángulo debe bajar de 65°
Esto asegura que el pico está "separado" de otros giros
Si nunca baja de 65°, son varios giros juntos, no una esquiva aislada
'''
# Prominencia mínima 
MIN_PROMINENCE_DEG = 20.0  # El pico debe sobresalir al menos 20° sobre el entorno  
# Suavizado
SMOOTH_SEC = 0.2 # Tiempo de suavizado para reducir ruido           
# Ventana de prominencia
PROM_WIN_SEC = 2.5  # Cuántos segundos **antes y después** del pico se miran para calcular la prominencia.      
'''
Ventana = 2.5s a cada lado del pico:

    |<--2.5s-->|<--2.5s-->|
              /\
    ____     /  \     ____
        \___/    \___/
'''
# Pendientes mínimas
MIN_RISE_RATE_DEG_PER_SEC = 25.0  # Pendiente mínima al subir 
MIN_FALL_RATE_DEG_PER_SEC = 25.0  # Pendiente mínima al bajar  

# Ancho del pico a nivel relativo
PEAK_WIDTH_LEVEL_FRAC = 0.70 # Nivel relativo para medir ancho (70% del pico)     
MAX_PEAK_WIDTH_SEC = 2.0   # Ancho máximo permitido a ese nivel       

'''
```
    100 |      /\         ← Altura del pico
     70 |    _/  \_       ← 70% de la altura (umbral)
     40 |   /      \
     10 |__/________\__
'''
MIN_SHARPNESS_DEG_PER_SEC = 40.0  # Relación entre la altura del pico y su ancho.
'''
PICO AGUDO (esquiva):
    100 |    /\        Sharpness = 100° / 1.5s = 66.7°/s 
     50 |   /  \
      0 |__/    \__
        |<1.5s>|

PICO ANCHO (curva):
    100 |   ___        Sharpness = 100° / 5s = 20°/s 
     50 | _/   \_
      0 |/       \___
        |<--5s-->|

Sharpness alta (> 40°/s) → Giro concentrado en poco tiempo →Esquiva 
Sharpness baja (< 40°/s) → Giro distribuido en mucho tiempo →Curva normal 
'''

USE_EXTRA_FILTERS = False # Habilitar filtros adicionales en P1
MIN_MAX_STEER_RATE_DEG = 20.0 # Velocidad máxima de giro mínima
MIN_MAX_TORQUE = 150.0 # Torque máximo mínimo

# --- FILTRO "PRE-MONTAÑA" (falsas esquivas) ---
BIG_MOUNTAIN_MIN_DEG = 150.0     # Si a la derecha hay una montaña >=150°, es la "grande"
PREMOUNTAIN_LOOKAHEAD_SEC = 4.0  # Cuánto miramos a la derecha
PREMOUNTAIN_MAX_GAP_SEC = 1.2    # Qué tan pegada debe empezar la montaña grande tras terminar la pequeña
PREMOUNTAIN_SMALL_MAX_DEG = 140.0  # La “pequeña” suele estar por debajo de ~140 (ajústalo si quieres)

'''
load_carstate(csv_path)
- Lee CSV.
- Comprueba que exista columna t (tiempo en nanosegundos).
- Crea timestamp_sec normalizando:(t - t[0]) / 1e9
- Asegura que existan columnas (si faltan, las crea como NaN):steeringAngleDeg, steeringRateDeg, steeringTorque, vEgo
- Ordena por tiempo y resetea índice. 

Resultado: Un DataFrame listo para analizar, con tiempo en segundos desde el inicio.'''

def load_carstate(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError(f"{csv_path} no tiene columna 't'")

    df["timestamp_sec"] = (df["t"] - df["t"].iloc[0]) / 1e9

    for c in ["steeringAngleDeg", "steeringRateDeg", "steeringTorque", "vEgo"]:
        if c not in df.columns:
            df[c] = np.nan

    return df.sort_values("timestamp_sec").reset_index(drop=True)

'''
event_duration_sec(df: pd.DataFrame, s: int, e: int) -> float: 
Calcula cuánto tiempo duró un evento (de inicio s a fin e).
Sirve para filtrar eventos:
- Si dura < 0.15s → Probablemente ruido
- Si dura > 8s → Probablemente no es una esquiva
'''
def event_duration_sec(df: pd.DataFrame, s: int, e: int) -> float:
    return float(df.loc[e, "timestamp_sec"] - df.loc[s, "timestamp_sec"])

'''
peak_abs_angle(angle_segment: np.ndarray) -> float
Encuentra el ángulo máximo (en valor absoluto) dentro de un segmento.

Sirve para saber qué tan extremo fue el giro:
- Peak = 30° → Giro suave
- Peak = 90° → Giro brusco (posible esquiva)

'''
def peak_abs_angle(angle_segment: np.ndarray) -> float:
    if angle_segment.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(angle_segment)))


def is_premountain_false_positive(df: pd.DataFrame, s: int, e: int) -> bool:
    """
    Devuelve True si la detección (s,e) parece una "montaña pequeña" que justo antes
    de una montaña grande (>=150°) y por tanto debe descartarse.
    """
    t = df["timestamp_sec"].to_numpy(dtype=float)
    a = np.abs(df["steeringAngleDeg"].to_numpy(dtype=float))

    if s < 0 or e >= len(df) or s >= e:
        return False

    # 1) pico de la detección actual (la "pequeña")
    small_peak = float(np.nanmax(a[s:e+1]))
    if not np.isfinite(small_peak):
        return False

    # Solo aplicamos el filtro a "montañas pequeñas" (si ya es enorme, no es premontaña)
    if small_peak > PREMOUNTAIN_SMALL_MAX_DEG:
        return False

    t_end = float(t[e])

    # 2) mirar a la derecha una ventana de tiempo
    right_mask = (t > t_end) & (t <= t_end + PREMOUNTAIN_LOOKAHEAD_SEC)
    idxs = np.where(right_mask)[0]
    if idxs.size == 0:
        return False

    # 3) si aparece una "montaña grande" en esa ventana
    # además exigimos que su "inicio" esté pegado al final de la pequeña (gap corto)
    # Definimos inicio como el primer instante donde supera ANGLE_ON_DEG.
    right_idxs = idxs
    right_angles = a[right_idxs]

    # ¿Hay algún punto >= BIG_MOUNTAIN_MIN_DEG?
    if float(np.nanmax(right_angles)) < BIG_MOUNTAIN_MIN_DEG:
        return False

    # Encontrar el primer cruce por encima de ANGLE_ON_DEG tras e
    # (inicio aproximado de la siguiente maniobra)
    start_big = None
    for k in right_idxs:
        if a[k] >= ANGLE_ON_DEG:
            start_big = k
            break

    if start_big is None:
        return False

    gap = float(t[start_big] - t_end)
    if gap > PREMOUNTAIN_MAX_GAP_SEC:
        return False

    return True


# 1) Eventos por histéresis
'''
Detecta todos los momentos donde el volante está significativamente girado.
¿Para qué sirve?
Es el primer paso para encontrar giros. Luego estos eventos se filtran para quedarse solo con esquivas.
'''
def detect_turn_events(df: pd.DataFrame) -> list[tuple[int, int]]:
    angle = df["steeringAngleDeg"].to_numpy(dtype=float)
    abs_a = np.abs(angle)

    events = []
    in_turn = False
    start = None

    for i in range(len(df)):
        if not in_turn:
            if abs_a[i] >= ANGLE_ON_DEG:
                in_turn = True
                start = i
        else:
            if abs_a[i] <= ANGLE_OFF_DEG:
                end = i
                events.append((start, end))
                in_turn = False
                start = None

    if in_turn and start is not None:
        events.append((start, len(df) - 1))

    return events

'''
Verifica si un evento detectado por histéresis es candidato a ser esquiva.
Criterios de filtrado:
Criterio        Umbral           Rechaza si...
Duración      0.15s - 4.5s     Muy corto (ruido) o muy largo (curva)
Ángulo pico    40° - 140°      Muy suave o físicamente imposible
Velocidad giro   > 20°/s       Giro muy lento
Torque           > 150 Nm  Sin fuerza aplicada

¿Para qué sirve?
Reducir falsos positivos. No todos los giros son esquivas.
'''
def passes_p1_filters(df: pd.DataFrame, s: int, e: int) -> bool:
    dur = event_duration_sec(df, s, e)
    if dur < P1_MIN_SUBTURN_SEC or dur > P1_MAX_SUBTURN_SEC:
        return False

    seg = df.iloc[s:e+1]
    a = seg["steeringAngleDeg"].to_numpy(dtype=float)
    p = peak_abs_angle(a)
    if not np.isfinite(p):
        return False
    if p < P1_PEAK_MIN_ABS_ANGLE_DEG or p > P1_PEAK_MAX_ABS_ANGLE_DEG:
        return False

    if USE_EXTRA_FILTERS:
        rate = seg["steeringRateDeg"].to_numpy(dtype=float)
        torque = seg["steeringTorque"].to_numpy(dtype=float)
        max_rate = np.nanmax(np.abs(rate))
        max_torque = np.nanmax(np.abs(torque))
        if max_rate < MIN_MAX_STEER_RATE_DEG and max_torque < MIN_MAX_TORQUE:
            return False

    return True


'''
Busca pares de giros que formen una maniobra evasiva completa.

Lógica del patrón:
Esquiva típica = GIRO1 (izq) + PAUSA + GIRO2 (der)

    40 |      /\              /\
    20 |     /  \            /  \
     0 |____/    \__________/    \____
        
        [GIRO 1]  [GAP]  [GIRO 2]
        
        |<------ EVASIÓN TOTAL ----->|

Criterios para emparejamiento:

Primer giro debe pasar passes_p1_filters()
Gap entre giros < 2 segundos
Duración total entre 1s y 8s
'''
def detect_evasions_first_peak_only(df: pd.DataFrame) -> list[dict]:
    events = detect_turn_events(df)
    if len(events) < 2:
        return []

    t = df["timestamp_sec"].to_numpy(dtype=float)

    evasions = []
    i = 0
    while i < len(events) - 1:
        s1, e1 = events[i]
        if not passes_p1_filters(df, s1, e1):
            i += 1
            continue

        found = False
        j = i + 1
        while j < len(events):
            s2, e2 = events[j]
            gap = float(t[s2] - t[e1])
            if gap > MAX_GAP_BETWEEN_PEAKS_SEC:
                break

            total = float(t[e2] - t[s1])
            if ENFORCE_TOTAL_DURATION and not (MIN_TOTAL_EVASION_SEC <= total <= MAX_TOTAL_EVASION_SEC):
                j += 1
                continue

            evasions.append({
                "method": "hysteresis_pair",
                "s_total": s1, "e_total": e2,
                "s1": s1, "e1": e1,
                "s2": s2, "e2": e2,
            })
            found = True
            break

        if found:
            i = j + 1
        else:
            i += 1

    return evasions


# 2) Detector de picos
'''
¿Qué hace? Aplica un suavizado (promedio móvil) a la señal.
Visualización:

ANTES del suavizado:
    80 |     /\  /\     ← Ruido, picos falsos
    40 |    /  \/  \
     0 |___/        \___

DESPUÉS del suavizado:
    80 |      ___        ← Curva suave, pico real
    40 |    _/   \_
     0 |___/       \___

¿Para qué sirve?
Eliminar vibraciones y ruido de sensores para detectar solo picos reales.
'''
def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")

'''
¿Qué hace?
Detecta picos aislados muy marcados que representan esquivas individuales (sin patrón de "ida y vuelta").

Proceso paso a paso:

| Paso | ¿Qué verifica? | Ejemplo |
|---|---|---|
| 1 | ¿Es máximo local? | `a[i] > a[i-1] AND a[i] > a[i+1]` |
| 2 | ¿Altura razonable? | 65° ≤ peak ≤ 180° |
| 3 | Buscar base del pico | Umbral = max(65°, peak×0.3) |
| 4 | ¿Duración válida? | 1.5s ≤ duración ≤ 5s |
| 5 | ¿Prominente? | Prominencia ≥ 20° |
| 6 | ¿Está aislado? | Antes/después < 65° |
| 7 | ¿Subida/bajada rápida? | Rate ≥ 17.5°/s (70% de 25) |
| 8 | ¿Pico estrecho? | Ancho @ 70% ≤ 2s |
| 9 | ¿Agudo? | Sharpness ≥ 40°/s |
'''
def detect_isolated_peaks_improved(df: pd.DataFrame) -> list[dict]:
    """
    Detecta picos aislados con parámetros más permisivos y mejor manejo de casos límite.
    """
    t = df["timestamp_sec"].to_numpy(dtype=float)
    a = np.abs(df["steeringAngleDeg"].to_numpy(dtype=float))

    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 0.05
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.05

    smooth_n = max(1, int(round(SMOOTH_SEC / dt)))
    prom_n = max(5, int(round(PROM_WIN_SEC / dt)))

    a_s = moving_average(a, smooth_n)

    peaks = []
    for i in range(1, len(a_s) - 1):
        # Máximo local
        if not (a_s[i] >= a_s[i - 1] and a_s[i] > a_s[i + 1]):
            continue

        # Altura del pico
        peak_val = float(a_s[i])
        if peak_val < PEAK_MIN_ABS_DEG or peak_val > PEAK_MAX_ABS_DEG:
            continue

        # Buscar límites usando umbral dinámico (% del pico) en vez de fijo
        threshold = max(BASELINE_MAX_DEG, peak_val * 0.3)  # 30% del pico o baseline
        
        left = i
        while left > 0 and a_s[left] > threshold:
            left -= 1

        right = i
        while right < len(a_s) - 1 and a_s[right] > threshold:
            right += 1

        # Duración del pico
        dur = float(t[right] - t[left])
        if dur < PEAK_MIN_DUR_SEC or dur > PEAK_MAX_DUR_SEC:
            continue

        l2 = max(0, i - prom_n)
        r2 = min(len(a_s) - 1, i + prom_n)
        
        # Excluir la zona del pico mismo para medir prominencia
        left_context = a_s[l2:left] if left > l2 else []
        right_context = a_s[right:r2+1] if right < r2 else []
        
        if len(left_context) > 0 and len(right_context) > 0:
            local_min = float(min(np.nanmin(left_context), np.nanmin(right_context)))
        elif len(left_context) > 0:
            local_min = float(np.nanmin(left_context))
        elif len(right_context) > 0:
            local_min = float(np.nanmin(right_context))
        else:
            local_min = 0.0
        
        prom = float(peak_val - local_min)
        if prom < MIN_PROMINENCE_DEG:
            continue

        # Solo verificar que los extremos estén "razonablemente bajos"
        # No exigir que toquen exactamente BASELINE_MAX_DEG
        margin = 3  # muestras de margen
        
        pre_vals = a_s[max(0, left-margin):left+1]
        post_vals = a_s[right:min(len(a_s), right+margin+1)]
        
        pre_ok = len(pre_vals) == 0 or float(np.nanmean(pre_vals)) < BASELINE_MAX_DEG * 1.2
        post_ok = len(post_vals) == 0 or float(np.nanmean(post_vals)) < BASELINE_MAX_DEG * 1.2
        
        if not (pre_ok and post_ok):
            continue

        # ===== CAMBIO 4: Agudeza más permisiva =====
        t_left = float(t[left])
        t_peak = float(t[i])
        t_right = float(t[right])

        rise_dt = max(1e-6, t_peak - t_left)
        fall_dt = max(1e-6, t_right - t_peak)

        base_left = float(a_s[left])
        base_right = float(a_s[right])

        rise_rate = (peak_val - base_left) / rise_dt
        fall_rate = (peak_val - base_right) / fall_dt

        # Solo verificar si hay ALGUNA pendiente significativa
        avg_rate = (rise_rate + fall_rate) / 2
        if avg_rate < MIN_RISE_RATE_DEG_PER_SEC * 0.7:  # 70% del umbral
            continue

        # Ancho del pico a nivel relativo
        level = PEAK_WIDTH_LEVEL_FRAC * peak_val

        wl = i
        while wl > 0 and a_s[wl] >= level:
            wl -= 1

        wr = i
        while wr < len(a_s) - 1 and a_s[wr] >= level:
            wr += 1

        width_sec = float(t[wr] - t[wl])
        if width_sec > MAX_PEAK_WIDTH_SEC:
            continue

        # Sharpness con umbral más bajo
        sharpness = peak_val / max(1e-6, width_sec)
        if sharpness < MIN_SHARPNESS_DEG_PER_SEC:
            continue

        peaks.append({
            "method": "isolated_peak",
            "s_total": left,
            "e_total": right,
            "s1": left, "e1": right,
            "s2": np.nan, "e2": np.nan,
            "peak_idx": i,
            "peak_abs_deg": peak_val,
            "peak_width_sec": width_sec,
            "rise_rate_deg_s": rise_rate,
            "fall_rate_deg_s": fall_rate,
            "sharpness_deg_s": sharpness,
            "prominence": prom,
            "duration_sec": dur,
        })

    # Deduplicación mejorada
    peaks_sorted = sorted(peaks, key=lambda d: d["peak_abs_deg"], reverse=True)
    merged = []
    
    for p in peaks_sorted:
        overlap = False
        for existing in merged:
            # Si hay solape significativo, quedarse con el de mayor amplitud
            if not (p["e_total"] < existing["s_total"] or p["s_total"] > existing["e_total"]):
                overlap = True
                break
        
        if not overlap:
            merged.append(p)
    
    # Reordenar por tiempo
    merged = sorted(merged, key=lambda d: d["s_total"])
    
    return merged


# 3) Union + deduplicación por solape
'''
¿Qué hace?
Calcula el **Intersection over Union (IoU)** en 1 dimensión (tiempo).

Fórmula: IoU = Intersección / Unión

Ejemplo visual:
Detección A:  [====]
Detección B:      [====]
              ----^^^^----
              
Intersección: ^^^^ (solapan)
Unión:       ----------- (todo el rango)

IoU = longitud_intersección / longitud_unión
'''
def iou_1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

'''
¿Qué hace?
Une detecciones que **probablemente son la misma esquiva** detectada por diferentes métodos.

*Ejemplo:
Método 1 (histéresis):  [========]
                         5s    8s

Método 2 (pico aislado):    [======]
                             6s   9s

IoU = (8-6)/(9-5) = 2/4 = 0.5 (50% solape)

Combinar en una sola: [==========]
                        5s        9s
                        method: "hysteresis_pair+peak"
'''
def merge_detections(df: pd.DataFrame, dets: list[dict]) -> list[dict]:
    """
    Unifica detecciones que solapan mucho (para no contar doble).
    """
    if not dets:
        return dets

    t = df["timestamp_sec"].to_numpy(dtype=float)
    dets = sorted(dets, key=lambda d: (d["s_total"], d["e_total"]))

    out = []
    for d in dets:
        s = int(d["s_total"])
        e = int(d["e_total"])
        if not out:
            out.append(d)
            continue

        ps = int(out[-1]["s_total"])
        pe = int(out[-1]["e_total"])

        # solape temporal en segundos
        iou = iou_1d(float(t[s]), float(t[e]), float(t[ps]), float(t[pe]))
        if iou >= 0.35:
            # combinar: ampliamos rango y marcamos método combinado
            out[-1]["s_total"] = min(ps, s)
            out[-1]["e_total"] = max(pe, e)
            out[-1]["method"] = out[-1]["method"] + "+peak"
        else:
            out.append(d)

    return out


# 4) Resumen / Plot
'''
¿Qué hace?
Crea una tabla resumen con las características de cada esquiva detectada.
Resultado (ejemplo):
video  evasion_id  method  t_start  t_end  duration_sec  max_abs_angle_deg  mean_vEgo
video_01  1  hysteresis_pair  5.2  7.8  2.68   5.34   5.2
video_01  2  isolated_peak   15.4   18.1   2.79   2.13   8.7
video_02   1 hysteresis_pair+peak    3.1    5.9    2.87    8.5    .3
'''
def summarize(df: pd.DataFrame, detections: list[dict], video_name: str) -> pd.DataFrame:
    rows = []
    for k, ev in enumerate(detections, start=1):
        sT, eT = int(ev["s_total"]), int(ev["e_total"])
        t0 = float(df.loc[sT, "timestamp_sec"])
        t1 = float(df.loc[eT, "timestamp_sec"])
        seg = df.iloc[sT:eT+1]

        a = seg["steeringAngleDeg"].to_numpy(dtype=float)
        rate = seg["steeringRateDeg"].to_numpy(dtype=float)
        torque = seg["steeringTorque"].to_numpy(dtype=float)
        vego = seg["vEgo"].to_numpy(dtype=float)

        rows.append({
            "video": video_name,
            "evasion_id": k,
            "method": ev.get("method", "unknown"),
            "t_start": t0,
            "t_end": t1,
            "duration_sec": t1 - t0,
            "max_abs_angle_deg": float(np.nanmax(np.abs(a))),
            "mean_abs_angle_deg": float(np.nanmean(np.abs(a))),
            "max_abs_steer_rate_deg": float(np.nanmax(np.abs(rate))),
            "mean_abs_steer_rate_deg": float(np.nanmean(np.abs(rate))),
            "max_abs_torque": float(np.nanmax(np.abs(torque))),
            "mean_abs_torque": float(np.nanmean(np.abs(torque))),
            "mean_vEgo": float(np.nanmean(vego)) if np.isfinite(np.nanmean(vego)) else np.nan,
            "min_vEgo": float(np.nanmin(vego)) if np.isfinite(np.nanmin(vego)) else np.nan,
        })

    return pd.DataFrame(rows)

def plot(df: pd.DataFrame, detections: list[dict], out_png: str, title: str):
    t = df["timestamp_sec"].to_numpy(dtype=float)
    angle = np.abs(df["steeringAngleDeg"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, angle, label="|steeringAngleDeg|", linewidth=1.0, alpha=0.9)
    ax.axhline(ANGLE_ON_DEG, linestyle="--", linewidth=1.5, label=f"ANGLE_ON={ANGLE_ON_DEG}")
    ax.axhline(ANGLE_OFF_DEG, linestyle="--", linewidth=1.0, label=f"ANGLE_OFF={ANGLE_OFF_DEG}")

    for ev in detections:
        sT, eT = int(ev["s_total"]), int(ev["e_total"])
        ax.axvspan(df.loc[sT, "timestamp_sec"], df.loc[eT, "timestamp_sec"], alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Grados")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csvs = [f for f in os.listdir(CARSTATE_DIR) if f.endswith(".csv") and "carState" in f]
    if not csvs:
        print(f"No encontré CSVs carState en: {CARSTATE_DIR}")
        return

    all_rows = []

    for fname in sorted(csvs):
        path = os.path.join(CARSTATE_DIR, fname)
        video_name = fname.replace("--qlog_carState.csv", "").replace("--rlog_carState.csv", "")

        try:
            df = load_carstate(path)
        except Exception as e:
            print(f"{fname}: error cargando ({e})")
            continue

        det_a = detect_evasions_first_peak_only(df)   
        det_b = detect_isolated_peaks_improved(df)           
        detections = merge_detections(df, det_a + det_b)

        # --- Filtrar falsas esquivas tipo "pre-montaña" ---
        detections = [
            d for d in detections
            if not is_premountain_false_positive(df, int(d["s_total"]), int(d["e_total"]))
        ]

        summary = summarize(df, detections, video_name)
        out_csv = os.path.join(OUT_DIR, f"{video_name}__turns.csv")
        summary.to_csv(out_csv, index=False)

        out_png = os.path.join(OUT_DIR, f"{video_name}__turns.png")
        plot(df, detections, out_png, title=f"{video_name} | esquivas detectadas: {len(summary)}")

        all_rows.append(summary)

        if len(summary) == 0:
            print(f"✓ {video_name}: 0 esquivas detectadas")
        else:
            print(f"✓ {video_name}: {len(summary)} esquivas | ej: {summary.loc[0,'t_start']:.1f}s–{summary.loc[0,'t_end']:.1f}s")

    if all_rows:
        global_df = pd.concat(all_rows, ignore_index=True)
        global_path = os.path.join(OUT_DIR, "_ALL_TURNS.csv")
        global_df.to_csv(global_path, index=False)
        print(f"\nGuardado resumen global: {global_path}")

    print(f"\nSalida en: {OUT_DIR}")

if __name__ == "__main__":
    main()

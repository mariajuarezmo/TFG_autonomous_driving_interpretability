# Importación de librerías necesarias
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ========== CONFIGURACIÓN ==========
base_data_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_datasets"
output_dir = r"C:\Users\maria\Escritorio\Personal\TFG\yoloVideo\pilotnet_processed"

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("PREPROCESAMIENTO DE IMÁGENES PARA PILOTNET")
print("=" * 70)


# ========== CARGA DE RUTAS Y TORQUES ==========
all_images = []
all_torques = []

print("\n[1/4] Cargando rutas de imágenes desde carpetas...")
for folder_name in os.listdir(base_data_dir):
    folder_path = os.path.join(base_data_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue
    
    data_txt = os.path.join(folder_path, "data.txt")
    if not os.path.exists(data_txt):
        continue
    
    with open(data_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) != 2:
                continue
            
            img_name, torque = parts
            img_path = os.path.join(folder_path, img_name)
            
            if os.path.exists(img_path):
                all_images.append(img_path)
                all_torques.append(float(torque))

print(f"   ✓ Total de imágenes encontradas: {len(all_images)}")


# ========== ANÁLISIS DE DISTRIBUCIÓN DE TORQUES ==========
print("\n[2/4] Analizando distribución de torques...")
torques_tensor = torch.tensor(all_torques, dtype=torch.float32)

min_torque = torques_tensor.min().item()
max_torque = torques_tensor.max().item()
mean_torque = torques_tensor.mean().item()
std_torque = torques_tensor.std().item()

print(f"\n Estadísticas originales:")
print(f"   Min:  {min_torque:8.2f}")
print(f"   Max:  {max_torque:8.2f}")
print(f"   Mean: {mean_torque:8.2f}")
print(f"   Std:  {std_torque:8.2f}")

# Contar positivos vs negativos
num_positivos = (torques_tensor > 0).sum().item()
num_negativos = (torques_tensor < 0).sum().item()
num_ceros = (torques_tensor == 0).sum().item()

print(f"\n Distribución de signos:")
print(f"   Positivos (derecha): {num_positivos:6d} ({num_positivos/len(all_torques)*100:.1f}%)")
print(f"   Negativos (izq):     {num_negativos:6d} ({num_negativos/len(all_torques)*100:.1f}%)")
print(f"   Ceros (recto):       {num_ceros:6d} ({num_ceros/len(all_torques)*100:.1f}%)")


# ========== NORMALIZACIÓN MEJORADA==========
print("\n[3/4] Normalizando torques")


# Normalización Min-Max a [-1, 1] (SIMÉTRICA)
# Esta opción garantiza que el rango sea exactamente [-1, 1]
range_torque = max_torque - min_torque
torques_normalized_v1 = 2 * (torques_tensor - min_torque) / range_torque - 1
print(f"\n MÉTODO 1: Normalización Min-Max simétrica")
print(f"   Rango: {range_torque:.2f}")
print(f"   Offset: {min_torque:.2f}")
print(f"   Rango normalizado: [{torques_normalized_v1.min():.3f}, {torques_normalized_v1.max():.3f}]")

torques_normalized = torques_normalized_v1
normalization_params = {
    'method': 'minmax_symmetric',
    'min_torque': min_torque,
    'max_torque': max_torque,
    'range': range_torque
}

# Verificar simetría
print(f"\n Verificación de simetría:")
print(f"   Positivos normalizados: max = {torques_normalized[torques_tensor > 0].max():.3f}")
print(f"   Negativos normalizados: min = {torques_normalized[torques_tensor < 0].min():.3f}")
print(f"   Balance: {abs(torques_normalized.max() + torques_normalized.min()):.6f} (debe ser ≈0)")


# ========== PROCESAMIENTO DE IMÁGENES ==========
print("\n[4/4] Procesando imágenes...")

transform = transforms.Compose([
    transforms.CenterCrop((66, 200)),
    transforms.ToTensor(),
])

images_tensor = []

for img_path in tqdm(all_images, desc="   Procesando"):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    images_tensor.append(img)

images_tensor = torch.stack(images_tensor)

print(f"\n   ✓ Tensor de imágenes: {images_tensor.shape}")
print(f"   ✓ Tensor de torques: {torques_normalized.shape}")


# ========== GUARDAR DATOS PROCESADOS ==========
print("\n[5/5] Guardando datos procesados...")

data_to_save = {
    'images': images_tensor,
    'torques': torques_normalized,
    'normalization_params': normalization_params,
    'original_stats': {
        'min': min_torque,
        'max': max_torque,
        'mean': mean_torque,
        'std': std_torque
    },
    'num_samples': len(all_images)
}

output_file = os.path.join(output_dir, "processed_data_v3.pt")
torch.save(data_to_save, output_file)

print(f"\n Datos guardados: {output_file}")
print(f" Tamaño: {os.path.getsize(output_file) / (1024**2):.2f} MB")

# Guardar información de configuración
config_file = os.path.join(output_dir, "preprocessing_config_v3.txt")
with open(config_file, "w") as f:
    f.write("CONFIGURACIÓN DEL PREPROCESAMIENTO (VERSIÓN MEJORADA)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Directorio origen: {base_data_dir}\n")
    f.write(f"Total de imágenes: {len(all_images)}\n")
    f.write(f"Dimensiones imagen: {list(images_tensor.shape[1:])}\n\n")
    
    f.write("ESTADÍSTICAS DE TORQUES ORIGINALES:\n")
    f.write(f"  Min:  {min_torque:8.2f}\n")
    f.write(f"  Max:  {max_torque:8.2f}\n")
    f.write(f"  Mean: {mean_torque:8.2f}\n")
    f.write(f"  Std:  {std_torque:8.2f}\n\n")
    
    f.write("DISTRIBUCIÓN DE SIGNOS:\n")
    f.write(f"  Positivos: {num_positivos} ({num_positivos/len(all_torques)*100:.1f}%)\n")
    f.write(f"  Negativos: {num_negativos} ({num_negativos/len(all_torques)*100:.1f}%)\n")
    f.write(f"  Ceros:     {num_ceros} ({num_ceros/len(all_torques)*100:.1f}%)\n\n")
    
    f.write("NORMALIZACIÓN APLICADA:\n")
    f.write(f"  Método: Min-Max Simétrica [-1, 1]\n")
    f.write(f"  Rango normalizado: [{torques_normalized.min():.3f}, {torques_normalized.max():.3f}]\n\n")
    
    f.write("FÓRMULAS DE CONVERSIÓN:\n")
    f.write("  Normalización:\n")
    f.write(f"    norm = 2 * (real - {min_torque:.2f}) / {range_torque:.2f} - 1\n\n")
    f.write("  Desnormalización:\n")
    f.write(f"    real = (norm + 1) * {range_torque:.2f} / 2 + {min_torque:.2f}\n\n")
    
    f.write("MÉTODO NORMALIZACIÓN:\n")
    f.write(f" (max abs):     [{torques_normalized_v1.min():.3f}, {torques_normalized_v1.max():.3f}] Asimétrico\n")



print(f" Configuración guardada: {config_file}")

# Crear función helper para desnormalizar
helper_file = os.path.join(output_dir, "denormalize_helper.py")
with open(helper_file, "w") as f:
    f.write("# Helper para desnormalizar predicciones\n")
    f.write("import torch\n\n")
    f.write(f"MIN_TORQUE = {min_torque}\n")
    f.write(f"MAX_TORQUE = {max_torque}\n")
    f.write(f"RANGE_TORQUE = {range_torque}\n\n")
    f.write("def denormalize(normalized_torque):\n")
    f.write('    """\n')
    f.write('    Convierte torque normalizado [-1, 1] a grados reales.\n')
    f.write('    \n')
    f.write('    Args:\n')
    f.write('        normalized_torque: Tensor o valor en rango [-1, 1]\n')
    f.write('    \n')
    f.write('    Returns:\n')
    f.write('        Torque en grados\n')
    f.write('    """\n')
    f.write(f"    return (normalized_torque + 1) * {range_torque} / 2 + {min_torque}\n\n")
    f.write("def normalize(real_torque):\n")
    f.write('    """\n')
    f.write('    Convierte torque en grados a normalizado [-1, 1].\n')
    f.write('    \n')
    f.write('    Args:\n')
    f.write('        real_torque: Torque en grados\n')
    f.write('    \n')
    f.write('    Returns:\n')
    f.write('        Torque normalizado [-1, 1]\n')
    f.write('    """\n')
    f.write(f"    return 2 * (real_torque - {min_torque}) / {range_torque} - 1\n\n")
    f.write("# Ejemplo de uso:\n")
    f.write("# pred_norm = model(image)  # Output: valor en [-1, 1]\n")
    f.write("# pred_grados = denormalize(pred_norm)\n")

print(f"   ✓ Helper de desnormalización: {helper_file}")

print("\n" + "=" * 70)
print("PREPROCESAMIENTO COMPLETADO")
print("=" * 70)
print(f"Archivo principal: processed_data_v3.pt")
print(f"Normalización: Min-Max simétrica [-1, 1]")
print(f"Rango real: [{min_torque:.0f}, {max_torque:.0f}]")
print("=" * 70)
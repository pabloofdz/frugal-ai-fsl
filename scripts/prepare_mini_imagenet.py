from tqdm import tqdm
import numpy as np
import shutil
import os

from config import DATA_PATH
from few_shot.utils import mkdir, rmdir

# Función para imprimir mensajes de depuración
def debug_message(message):
    print(f"[DEBUG] {message}")

# Limpieza de carpetas antiguas
debug_message("Limpieza de carpetas antiguas")
rmdir(DATA_PATH + '/miniImageNet/images_background')
rmdir(DATA_PATH + '/miniImageNet/images_evaluation')
mkdir(DATA_PATH + '/miniImageNet/images_background')
mkdir(DATA_PATH + '/miniImageNet/images_evaluation')

# Encontrar identidades de clases
debug_message("Encontrando identidades de clases")
classes = []
for root, dirs, files in os.walk(DATA_PATH + '/miniImageNet/images/'):
    for d in dirs:
        classes.append(d)
    break  # Solo necesitamos el primer nivel de subdirectorios

debug_message(f"Clases encontradas: {classes[:5]}... (total: {len(classes)})")
classes = list(set(classes))

# División en conjuntos de entrenamiento y evaluación
debug_message("Dividiendo en clases de entrenamiento y evaluación")
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:80], classes[80:]
debug_message(f"Clases de entrenamiento: {background_classes[:5]}... (total: {len(background_classes)})")
debug_message(f"Clases de evaluación: {evaluation_classes[:5]}... (total: {len(evaluation_classes)})")

# Creación de carpetas de clases
debug_message("Creando carpetas para clases de entrenamiento")
for c in background_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_background/{c}/')

debug_message("Creando carpetas para clases de evaluación")
for c in evaluation_classes:
    mkdir(DATA_PATH + f'/miniImageNet/images_evaluation/{c}/')

# Mover imágenes a la ubicación correcta
debug_message("Moviendo imágenes a la ubicación correcta")
for root, dirs, files in os.walk(DATA_PATH + '/miniImageNet/images'):
    for f in tqdm(files, total=600*100):
        if f.endswith('.JPEG'):
            class_name = os.path.basename(root)
            # Send to correct folder
            subset_folder = 'images_evaluation' if class_name in evaluation_classes else 'images_background'
            src = os.path.join(root, f)
            dst = os.path.join(DATA_PATH, f'miniImageNet/{subset_folder}/{class_name}/{f}')
            debug_message(f"Copiando {src} a {dst}")
            shutil.copy(src, dst)

debug_message("Script completado")

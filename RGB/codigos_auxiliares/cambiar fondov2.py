import os
import cv2
import numpy as np

INPUT_FOLDER  = "input_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Parámetros ajustables
BRIGHT_THRESH = 220    # píxeles con gris >= este valor se consideran "claros"
MIN_FRUIT_AREA = 5000  # eliminar componentes más pequeñas (ajusta según resolución)
FEATHER_RADIUS = 2    # radio para suavizar bordes (en píxeles)
KEEP_ONLY_LARGEST = False  # True si quieres conservar sólo la componente mayor

def detect_background_by_border(gray, bright_thresh):
    # 1) detectar píxeles claros
    bright_mask = (gray >= bright_thresh).astype(np.uint8) * 255

    # 2) obtener componentes conectadas de las zonas claras
    num_labels, labels = cv2.connectedComponents(bright_mask, connectivity=8)

    # 3) identificar componentes que tocan los bordes -> verdadero fondo
    h, w = gray.shape
    border_labels = set()
    # top & bottom rows
    border_labels.update(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[h-1, :]))
    # left & right cols
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, w-1]))

    # 4) construir máscara de fondo (1 donde es fondo)
    bg_mask = np.zeros_like(bright_mask, dtype=np.uint8)
    for lab in border_labels:
        if lab == 0:
            continue
        bg_mask[labels == lab] = 255

    return bg_mask

def keep_largest_components(binary_mask, min_area=0, keep_only_largest=False):
    # binary_mask: 0/255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask)

    out_mask = np.zeros_like(binary_mask)
    areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            areas.append((i, area))

    if not areas:
        return out_mask

    if keep_only_largest:
        # conservar sólo la mayor
        largest = max(areas, key=lambda x: x[1])[0]
        out_mask[labels == largest] = 255
    else:
        # conservar todas que superen min_area
        for lab, _ in areas:
            out_mask[labels == lab] = 255

    return out_mask

def feather_mask(mask, radius):
    # suavizado por blur en máscara normalizada
    if radius <= 0:
        return (mask > 127).astype(np.uint8) * 255
    float_mask = mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(float_mask, (radius*2+1, radius*2+1), 0)
    return (blurred * 255).astype(np.uint8)

for fname in sorted(os.listdir(INPUT_FOLDER)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path_in = os.path.join(INPUT_FOLDER, fname)
    img = cv2.imread(path_in)
    if img is None:
        print(f"❌ No se pudo leer {fname}")
        continue

    # 0) trabajar en copia a tamaño original
    h, w = img.shape[:2]

    # 1) gris y detectar fondo por borde
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg_by_border = detect_background_by_border(gray, BRIGHT_THRESH)

    # 2) fondo final = bg_by_border (puedes cerrar pequeños huecos)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    bg_clean = cv2.morphologyEx(bg_by_border, cv2.MORPH_CLOSE, kernel)
    bg_clean = cv2.morphologyEx(bg_clean, cv2.MORPH_OPEN, kernel)

    # 3) fruta_mask = inverse of background
    fruit_mask = cv2.bitwise_not(bg_clean)

    # 4) limpiar y quedarnos con componentes grandes (opcional: solo la mayor)
    fruit_mask = keep_largest_components(fruit_mask, min_area=MIN_FRUIT_AREA, keep_only_largest=KEEP_ONLY_LARGEST)

    # 5) suavizar bordes
    fruit_mask_feather = feather_mask(fruit_mask, FEATHER_RADIUS)  # 0-255 float mask

    # 6) componer sobre fondo negro con blending usando máscara suavizada
    alpha = (fruit_mask_feather.astype(np.float32) / 255.0)[:,:,None]  # shape HxWx1
    fg = img.astype(np.float32)
    bg = np.zeros_like(img).astype(np.float32)  # negro
    composed = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)

    # 7) guardar
    out_path = os.path.join(OUTPUT_FOLDER, fname)
    cv2.imwrite(out_path, composed)
    print(f"✔ {fname} procesada -> {out_path}")

print("\n🎉 Listo. Revisa la carpeta:", OUTPUT_FOLDER)

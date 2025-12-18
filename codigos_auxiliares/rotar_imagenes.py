import os
import cv2

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NUM_IMAGES = 10
ANGLE_STEP = 36  # grados

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_FOLDER, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ No se pudo leer {filename}")
        continue

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    name, ext = os.path.splitext(filename)

    for i in range(NUM_IMAGES):
        angle = i * ANGLE_STEP

        # Matriz de rotación
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotar imagen
        rotated = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        out_name = f"{name}_rot_{angle}{ext}"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        cv2.imwrite(out_path, rotated)

    print(f"✔ Procesada: {filename}")

print("🎉 Rotación finalizada. Revisa la carpeta output_images")

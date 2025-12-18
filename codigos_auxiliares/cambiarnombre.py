import os
import shutil

INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

contador = 1

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    ext = os.path.splitext(filename)[1]  # extensión original
    new_name = f"{contador} (mala){ext}"

    src_path = os.path.join(INPUT_FOLDER, filename)
    dst_path = os.path.join(OUTPUT_FOLDER, new_name)

    shutil.copy2(src_path, dst_path)
    print(f"✔ {filename} -> {new_name}")

    contador += 1

print("🎉 Proceso terminado. Revisa la carpeta output_images")

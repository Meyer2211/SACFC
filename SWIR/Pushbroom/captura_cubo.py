import time
import json
import msvcrt
from pathlib import Path

import gxipy as gx
import numpy as np
import matplotlib.pyplot as plt


def save_png_uint8(img: np.ndarray, out_path: Path, title: str = ""):
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())

    if mx <= mn:
        img8 = np.zeros_like(img, dtype=np.uint8)
    else:
        img8 = ((img - mn) * (255.0 / (mx - mn))).clip(0, 255).astype(np.uint8)

    plt.figure(figsize=(8, 5))
    plt.imshow(img8, cmap="gray")
    plt.colorbar(label="Intensidad escalada 8-bit")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("PNG guardado:", out_path)


def set_exposure_time(cam, exposure_time_us: float):
    remote = cam.get_remote_device_feature_control()

    if remote.is_implemented("ExposureAuto") and remote.is_writable("ExposureAuto"):
        remote.get_enum_feature("ExposureAuto").set("Off")

    if remote.is_implemented("ExposureTime") and remote.is_writable("ExposureTime"):
        remote.get_float_feature("ExposureTime").set(float(exposure_time_us))

    print(f"Tiempo de exposición configurado a {exposure_time_us} us")


def capture_until_key(
    output_dir: str = "cubos_prueba",
    exposure_time_us: float = 8000,
    camera_index: int = 1,
    stop_key: str = "q",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    dm = gx.DeviceManager()
    num, info = dm.update_all_device_list()

    print("Cámaras detectadas:", num)

    if num == 0:
        raise SystemExit("No detecté cámara. Revisa Galaxy Viewer / red / firewall.")

    cam = dm.open_device_by_index(camera_index)

    frames = []

    try:
        cam.stream_on()
        time.sleep(0.3)

        set_exposure_time(cam, exposure_time_us)

        print("\nCapturando continuamente...")
        print(f"Presione '{stop_key}' para detener y guardar el cubo.\n")

        t0 = time.time()
        z = 0

        while True:
            raw = cam.data_stream[0].get_image()

            if raw is None:
                print(f"Frame {z}: None, se omite")
                continue

            img = raw.get_numpy_array()

            if img is None:
                print(f"Frame {z}: no convertido a numpy, se omite")
                continue

            frames.append(img.copy())
            z += 1

            # Detener con tecla
            if msvcrt.kbhit():
                key = msvcrt.getch().decode(errors="ignore").lower()
                if key == stop_key.lower():
                    print(f"\nTecla '{stop_key}' presionada. Deteniendo captura...")
                    break

        if len(frames) == 0:
            print("No se capturaron frames.")
            return

        t1 = time.time()
        elapsed = t1 - t0
        fps = len(frames) / elapsed

        print("\nCaptura terminada")
        print(f"Frames capturados: {len(frames)}")
        print(f"Tiempo total: {elapsed:.2f} s")
        print(f"FPS promedio: {fps:.2f}")

        # Cubo Y, X, Z
        cube = np.stack(frames, axis=2)

        print("Shape cubo:", cube.shape)
        print("Dtype cubo:", cube.dtype)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        cube_path = out_dir / f"cube_{timestamp}.npy"
        np.save(cube_path, cube)
        print("Cubo guardado:", cube_path)

        n_frames = cube.shape[2]

        selected_frames = {
            "first" : 0,
            "middle": n_frames // 2,
            "last"  : n_frames - 1,
        }

        for name, idx in selected_frames.items():
            frame    = cube[:, :, idx]
            png_path = out_dir / f"cube_{timestamp}_{name}_frame_{idx:04d}.png"
            save_png_uint8(
                frame, png_path,
                title=f"{name} frame | z={idx} | exposure={exposure_time_us} us",
            )

        mean_img      = np.mean(cube.astype(np.float32), axis=2)
        mean_png_path = out_dir / f"cube_{timestamp}_mean.png"
        save_png_uint8(
            mean_img, mean_png_path,
            title=f"Promedio del cubo | {n_frames} frames",
        )

        metadata = {
            "cube_file"       : str(cube_path),
            "shape"           : list(cube.shape),
            "dtype"           : str(cube.dtype),
            "n_frames"        : int(n_frames),
            "height"          : int(cube.shape[0]),
            "width"           : int(cube.shape[1]),
            "exposure_time_us": exposure_time_us,
            "elapsed_s"       : elapsed,
            "fps_average"     : fps,
            "axis_order"      : "Y, X, Z",
            "stop_key"        : stop_key,
        }

        metadata_path = out_dir / f"cube_{timestamp}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print("Metadata guardada:", metadata_path)

    finally:
        cam.stream_off()
        cam.close_device()
        print("Cámara cerrada correctamente.")


if __name__ == "__main__":
    capture_until_key(
        output_dir=r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\cubos_prueba",
        exposure_time_us=4000,
        camera_index=1,
        stop_key="q",
    )

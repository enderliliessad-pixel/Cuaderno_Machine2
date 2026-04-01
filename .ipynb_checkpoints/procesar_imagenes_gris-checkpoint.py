import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def procesar_imagen(ruta_entrada, ruta_salida, size):
    try:
        img = Image.open(ruta_entrada).convert("L")  # 🔥 escala de grises
        img = img.resize(size)

        img_array = np.array(img, dtype=np.float32) / 255.0  # 🔥 normalización

        # agregar canal (224,224) → (224,224,1)
        img_array = np.expand_dims(img_array, axis=-1)

        np.save(ruta_salida, img_array)

        return True
    except:
        return False


def procesar_dataset(input_root, output_root, size):
    total = 0
    errores = 0

    for split in ["train", "val"]:
        input_split = os.path.join(input_root, split)
        output_split = os.path.join(output_root, split)

        os.makedirs(output_split, exist_ok=True)

        for clase in os.listdir(input_split):
            input_clase = os.path.join(input_split, clase)
            output_clase = os.path.join(output_split, clase)

            os.makedirs(output_clase, exist_ok=True)

            archivos = os.listdir(input_clase)

            for archivo in tqdm(archivos, desc=f"{split} - {clase}"):
                ruta_in = os.path.join(input_clase, archivo)
                nombre_salida = os.path.splitext(archivo)[0] + ".npy"
                ruta_out = os.path.join(output_clase, nombre_salida)

                ok = procesar_imagen(ruta_in, ruta_out, size)

                if ok:
                    total += 1
                else:
                    errores += 1

    print("\n=== RESUMEN ===")
    print(f"Procesadas: {total}")
    print(f"Errores: {errores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)

    args = parser.parse_args()

    procesar_dataset(
        args.input,
        args.output,
        (args.width, args.height)
    )
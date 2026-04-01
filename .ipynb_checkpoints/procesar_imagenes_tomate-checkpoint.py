import argparse
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True
VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def process_image(src_path: Path, dst_path: Path, size=(224, 224), save_mode='png'):
    with Image.open(src_path) as img:
        img = img.convert('RGB')
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalizado 0-1

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if save_mode == 'npy':
        np.save(dst_path.with_suffix('.npy'), arr)
    else:
        # guardar como imagen estándar ya redimensionada
        out = Image.fromarray((arr * 255).astype(np.uint8))
        if save_mode == 'jpg':
            out.save(dst_path.with_suffix('.jpg'), quality=95)
        else:
            out.save(dst_path.with_suffix('.png'))


def scan_and_process(input_root: Path, output_root: Path, size=(224, 224), save_mode='png'):
    summary = {
        'input_root': str(input_root),
        'output_root': str(output_root),
        'size': list(size),
        'save_mode': save_mode,
        'processed': 0,
        'failed': 0,
        'classes': {}
    }

    for split in ['train', 'val']:
        split_dir = input_root / split
        if not split_dir.exists():
            continue

        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            class_name = class_dir.name
            summary['classes'].setdefault(split, {})[class_name] = 0

            for file_path in class_dir.rglob('*'):
                if file_path.suffix.lower() not in VALID_EXTS:
                    continue
                rel = file_path.relative_to(input_root)
                dst = output_root / rel
                try:
                    process_image(file_path, dst, size=size, save_mode=save_mode)
                    summary['processed'] += 1
                    summary['classes'][split][class_name] += 1
                except Exception as e:
                    summary['failed'] += 1
                    print(f'[ERROR] {file_path}: {e}')

    return summary


def main():
    parser = argparse.ArgumentParser(description='Procesa y normaliza imágenes de tomate.')
    parser.add_argument('--input', required=True, help='Ruta de la carpeta raíz, por ejemplo: recursos')
    parser.add_argument('--output', required=True, help='Ruta de salida, por ejemplo: procesado')
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--format', choices=['png', 'jpg', 'npy'], default='png',
                        help='png/jpg guarda imágenes redimensionadas; npy guarda arrays normalizados 0-1')
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    summary = scan_and_process(
        input_root=input_root,
        output_root=output_root,
        size=(args.width, args.height),
        save_mode=args.format,
    )

    summary_path = output_root / 'resumen_procesamiento.json'
    output_root.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print('\n=== RESUMEN ===')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nResumen guardado en: {summary_path}')


if __name__ == '__main__':
    main()

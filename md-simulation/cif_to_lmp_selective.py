#!/usr/bin/env python3
"""
Конвертирует .vasp / .cif файлы из папки mp_init/ в LAMMPS data-файлы (.lmp).
Выходные файлы сохраняются в папку lammps_init/.

Использование:
  python convert_mp.py [<материал> [<nx> <ny> <nz>]]

Примеры:
  python convert_mp.py                # обработать все файлы (суперячейка 2x2x4)
  python convert_mp.py BN             # только материал BN, суперячейка 2x2x4
  python convert_mp.py BN 3 3 5       # BN с суперячейкой 3x3x5
"""

from pathlib import Path
import sys

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData

# --- Настройки по умолчанию ---
DEFAULT_SUPERCELL = (2, 2, 4)

# Директории относительно этого скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "mp_init"
OUTPUT_DIR = SCRIPT_DIR / "lammps_init"


def main() -> None:
    # --- Разбор аргументов командной строки ---
    if len(sys.argv) == 1:
        # Без аргументов: обрабатываем все файлы с суперячейкой по умолчанию
        target_materials = None
        supercell_dims = DEFAULT_SUPERCELL
    else:
        material_name = sys.argv[1]
        # Простейшая защита от случайного указания числа вместо имени
        if material_name.isdigit():
            print("Ошибка: первым аргументом должно быть название материала (не число).")
            sys.exit(1)
        target_materials = [material_name]

        if len(sys.argv) >= 5:
            # Пользователь передал три числа для размеров суперячейки
            try:
                nx, ny, nz = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
                supercell_dims = (nx, ny, nz)
            except ValueError:
                print("Ошибка: размеры суперячейки должны быть целыми числами, например: 3 3 5")
                sys.exit(1)
        elif len(sys.argv) > 2:
            # Аргументы есть, но не три числа
            print("Ошибка: укажите ровно три целых числа для суперячейки (nx ny nz) или опустите их для значений по умолчанию.")
            sys.exit(1)
        else:
            # Только название материала
            supercell_dims = DEFAULT_SUPERCELL

    # --- Сбор файлов ---
    OUTPUT_DIR.mkdir(exist_ok=True)
    extensions = (".vasp", ".cif")
    all_files = []
    for ext in extensions:
        all_files.extend(INPUT_DIR.glob(f"*{ext}"))

    if not all_files:
        print(f"Не найдено файлов .vasp или .cif в {INPUT_DIR}")
        sys.exit(1)

    if target_materials is not None:
        # Ищем точное совпадение по имени файла без расширения
        requested = target_materials[0]
        matching = [f for f in all_files if f.stem == requested]
        if not matching:
            print(f"Файл для материала '{requested}' не найден в {INPUT_DIR}")
            print(f"Доступные материалы: {', '.join(sorted(f.stem for f in all_files))}")
            sys.exit(1)
        files_to_process = matching
    else:
        files_to_process = all_files

    print(f"Найдено файлов для конвертации: {len(files_to_process)}")
    print(f"Размер суперячейки: {supercell_dims[0]}×{supercell_dims[1]}×{supercell_dims[2]}\n")

    # --- Обработка каждого файла ---
    for filepath in files_to_process:
        name = filepath.stem
        print(f"Обрабатываю: {name} ({filepath.name})")

        try:
            structure = Structure.from_file(filepath)

            # Создание суперячейки, если хотя бы одна размерность > 1
            if any(d > 1 for d in supercell_dims):
                structure.make_supercell(list(supercell_dims))

            lmp_data = LammpsData.from_structure(structure)

            # Безопасное имя файла (без скобок и пробелов)
            safe_name = name.replace("(", "").replace(")", "").replace(" ", "_")
            output_path = OUTPUT_DIR / f"{safe_name}.lmp"

            lmp_data.write_file(str(output_path))
            print(f"  → Сохранён {output_path.name}")

        except Exception as e:
            print(f"  ✗ Ошибка: {e}")

    print("\nГотово.")


if __name__ == "__main__":
    main()

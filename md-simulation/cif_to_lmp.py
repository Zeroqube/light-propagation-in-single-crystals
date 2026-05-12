"""
Конвертирует все .vasp / .cif файлы из папки mp_init/ в LAMMPS data-файлы (.lmp)
Выходные файлы сохраняются в папку lammps_init/
Пути определяются относительно расположения этого скрипта.
"""

from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
import sys

# --- Настройки ---
SUPERCELL_DIMS = (5, 5, 5)   # мультипликаторы дублирования ячейки (можно изменить)

# Директории относительно этого скрипта
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "mp_init"
OUTPUT_DIR = SCRIPT_DIR / "lammps_init"

# Создаём выходную папку, если её нет
OUTPUT_DIR.mkdir(exist_ok=True)

# Собираем все файлы с нужными расширениями
extensions = (".vasp", ".cif")
files = []
for ext in extensions:
    files.extend(INPUT_DIR.glob(f"*{ext}"))

if not files:
    print(f"Не найдено файлов .vasp или .cif в {INPUT_DIR}")
    sys.exit(1)

print(f"Найдено {len(files)} файлов для конвертации.\n")

for filepath in files:
    name = filepath.stem  # имя без расширения
    print(f"Обрабатываю: {name} ({filepath.name})")

    try:
        # Чтение структуры (поддерживает POSCAR и CIF)
        structure = Structure.from_file(filepath)

        # Создание суперячейки
        if any(d > 1 for d in SUPERCELL_DIMS):
            structure.make_supercell(list(SUPERCELL_DIMS))

        # Генерация LAMMPS data
        lmp_data = LammpsData.from_structure(structure)

        # Безопасное имя файла (убираем скобки и пробелы)
        safe_name = name.replace("(", "").replace(")", "").replace(" ", "_")
        output_path = OUTPUT_DIR / f"{safe_name}.lmp"

        lmp_data.write_file(str(output_path))
        print(f"  → Сохранён {output_path.name}")

    except Exception as e:
        print(f"  ✗ Ошибка: {e}")

print("\nГотово.")

"""
Конвертирует все .vasp / .cif файлы из папки mp_init/ в LAMMPS data-файлы (.lmp).
Автоматически устраняет наклон (tilt) ячейки для гексагональных и тригональных структур.
Выходные файлы сохраняются в папку lammps_init/
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = SCRIPT_DIR / "mp_init"
OUTPUT_DIR = SCRIPT_DIR / "lammps_init"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Параметры суперячейки
# ---------------------------------------------------------------------------
# Целевое число атомов в суперячейке (приблизительно).
# Скрипт подбирает кратности автоматически.
TARGET_ATOMS = 500

# Для ортогонализированных гекс./триг. структур атомов после M удваивается,
# поэтому цель делится на 2 перед масштабированием.
ORTHO_MATRIX = [[1, 0, 0],
                [1, 2, 0],   # b_new = a_old + 2*b_old → xy = 0
                [0, 0, 1]]

# Сингонии, для которых нужна ортогонализация
NEEDS_ORTHO = {"hexagonal", "trigonal"}

# Ручные переопределения кратностей (safe_name → (nx, ny, nz)).
# Заполни при необходимости; иначе используется автоподбор.
MANUAL_SCALE: dict[str, tuple[int, int, int]] = {
    # "TiO2": (4, 4, 3),
}

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def lammps_tilt(lattice) -> dict[str, tuple[float, float]]:
    """
    Возвращает словарь нарушений LAMMPS-условий на tilt-факторы:
        |xy| <= lx/2,  |xz| <= lx/2,  |yz| <= ly/2

    Ключи: 'xy', 'xz', 'yz'.  Значение: (|tilt|, limit).
    Если нарушений нет — пустой словарь.
    """
    a_vec = lattice.matrix[0]
    b_vec = lattice.matrix[1]
    c_vec = lattice.matrix[2]

    lx = float(np.linalg.norm(a_vec))
    xy = float(np.dot(b_vec, a_vec / lx))
    ly = float(np.sqrt(max(np.dot(b_vec, b_vec) - xy**2, 0.0)))
    xz = float(np.dot(c_vec, a_vec / lx))
    yz = float((np.dot(b_vec, c_vec) - xy * xz) / ly) if ly > 1e-10 else 0.0

    violations: dict[str, tuple[float, float]] = {}
    if abs(xy) > lx / 2 + 1e-6:
        violations["xy"] = (abs(xy), lx / 2)
    if abs(xz) > lx / 2 + 1e-6:
        violations["xz"] = (abs(xz), lx / 2)
    if abs(yz) > ly / 2 + 1e-6:
        violations["yz"] = (abs(yz), ly / 2)
    return violations


def auto_scale(n_atoms_now: int, target: int,
               anisotropy: float = 1.0) -> tuple[int, int, int]:
    """
    Подбирает (nx, ny, nz) так чтобы итоговое число атомов было близко к target.
    anisotropy > 1 означает что c-ось длиннее (вытянутая ячейка) —
    тогда nc ставится меньше.

    Простая эвристика: nx = ny = nz = round(cbrt(target / n_atoms_now)),
    с корректировкой nc для анизотропных ячеек.
    """
    ratio = target / max(n_atoms_now, 1)
    n = max(1, round(ratio ** (1 / 3)))
    nc = max(1, round(n / anisotropy))
    return (n, n, nc)


def crystal_anisotropy(structure: Structure) -> float:
    """Примерная анизотропия c/a для подбора nc."""
    a = structure.lattice.a
    c = structure.lattice.c
    return max(c / a, 1.0)


# ---------------------------------------------------------------------------
# Основная функция обработки одного файла
# ---------------------------------------------------------------------------

def process(filepath: Path) -> None:
    name     = filepath.stem
    safe     = name.replace("(", "").replace(")", "").replace(" ", "_")
    print(f"\n{'─'*55}")
    print(f"  {name}  ({filepath.suffix})")
    print(f"{'─'*55}")

    # 1. Читаем и стандартизируем структуру
    structure = Structure.from_file(filepath)
    sga = SpacegroupAnalyzer(structure)
    structure = sga.get_conventional_standard_structure()

    crystal_system = sga.get_crystal_system()
    spacegroup     = sga.get_space_group_symbol()
    n_prim         = len(structure)

    print(f"  Сингония:      {crystal_system}  ({spacegroup})")
    print(f"  Атомов в ячейке: {n_prim}")
    print(f"  a={structure.lattice.a:.3f}  b={structure.lattice.b:.3f}  "
          f"c={structure.lattice.c:.3f} Å")
    print(f"  α={structure.lattice.alpha:.1f}°  "
          f"β={structure.lattice.beta:.1f}°  "
          f"γ={structure.lattice.gamma:.1f}°")

    # 2. Ортогонализация для гексагональных / тригональных
    ortho_applied = False
    if crystal_system in NEEDS_ORTHO:
        structure.make_supercell(ORTHO_MATRIX)
        ortho_applied = True
        print(f"  Ортогонализация: матрица {ORTHO_MATRIX[1]} применена")
        print(f"  После ортогонализации: {len(structure)} атомов, "
              f"γ≈{structure.lattice.gamma:.1f}°")

    # 3. Масштабирование до TARGET_ATOMS
    if safe in MANUAL_SCALE:
        nx, ny, nz = MANUAL_SCALE[safe]
        print(f"  Масштаб: ручной {(nx, ny, nz)}")
    else:
        anis   = crystal_anisotropy(structure)
        target = TARGET_ATOMS // 2 if ortho_applied else TARGET_ATOMS
        nx, ny, nz = auto_scale(len(structure), target, anisotropy=anis)
        print(f"  Масштаб: авто {(nx, ny, nz)}  (анизотропия c/a ≈ {anis:.2f})")

    structure.make_supercell([nx, ny, nz])
    n_final = len(structure)

    # 4. Проверяем tilt после всех трансформаций
    violations = lammps_tilt(structure.lattice)
    if violations:
        # Попытка исправить наклон увеличением lx
        # (добавляем ещё одно повторение по x)
        print(f"  ⚠ Tilt нарушения: {violations} — увеличиваю nx на 1")
        structure.make_supercell([2, 1, 1])
        violations = lammps_tilt(structure.lattice)
        n_final = len(structure)

    if violations:
        print(f"  ✗ Tilt всё ещё нарушен: {violations}")
        print(f"    Добавь '{safe}' в MANUAL_SCALE с бо́льшим nx.")
    else:
        print(f"  ✓ Tilt OK")

    print(f"  Итого атомов:  {n_final}")
    print(f"  Параметры суперячейки: "
          f"a={structure.lattice.a:.2f}  "
          f"b={structure.lattice.b:.2f}  "
          f"c={structure.lattice.c:.2f} Å")

    # 5. Запись LAMMPS data-файла
    lmp_data   = LammpsData.from_structure(structure)
    out_path   = OUTPUT_DIR / f"{safe}.lmp"
    lmp_data.write_file(str(out_path))
    print(f"  → {out_path.name}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    files: list[Path] = []
    for ext in (".vasp", ".cif"):
        files.extend(INPUT_DIR.glob(f"*{ext}"))

    if not files:
        print(f"Не найдено файлов .vasp / .cif в {INPUT_DIR}")
        sys.exit(1)

    print(f"Найдено файлов: {len(files)}")
    print(f"Целевое число атомов: {TARGET_ATOMS}")

    ok, fail = [], []
    for fp in sorted(files):
        try:
            process(fp)
            ok.append(fp.stem)
        except Exception as e:
            print(f"  ✗ Ошибка ({fp.name}): {e}")
            fail.append(fp.stem)

    print(f"\n{'═'*55}")
    print(f"Готово: {len(ok)} OK,  {len(fail)} ошибок.")
    if fail:
        print(f"Ошибки: {', '.join(fail)}")


if __name__ == "__main__":
    main()

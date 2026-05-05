#!/usr/bin/env python3
"""
Генерирует LAMMPS input-скрипты в md-simulation/inputs/in.<material>
из реестра потенциалов potentials/potentials.csv и шаблона inputs/in.template.

Использование:
  python generate_inputs.py [<материал1> <материал2> ...]

Если аргументы не указаны, генерируются скрипты для всех материалов из CSV.
Пример: python generate_inputs.py BN
"""

from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

# Атомные массы (а.е.м.) для элементов, которые встречаются в текущих 39 материалах.
# Расширяй при добавлении новых систем.
ATOMIC_MASSES: dict[str, float] = {
    "H": 1.008, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.95, "Ag": 107.868, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Cs": 132.905, "Ba": 137.327,
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930,
    "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967, "Hf": 178.49,
    "Ta": 180.948, "W": 183.84, "Pt": 195.084, "Au": 196.967, "Hg": 200.592,
    "Pb": 207.2, "Bi": 208.980,
}

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "potentials" / "potentials.csv"
TEMPLATE_PATH = SCRIPT_DIR / "inputs" / "in.template"
LAMMPS_INIT_DIR = SCRIPT_DIR / "lammps_init"
INPUTS_DIR = SCRIPT_DIR / "inputs"
POTENTIALS_DIR = SCRIPT_DIR / "potentials"

STATUS_PRIORITY = {"kim": 0, "local": 1}


def safe_material_name(name: str) -> str:
    """LuAl3(BO3)4 -> LuAl3BO34 — как в lammps_init/."""
    return name.replace("(", "").replace(")", "").replace(" ", "_")


def parse_species_from_lmp(lmp_path: Path) -> list[str]:
    """Возвращает список элементов в порядке атомных типов LAMMPS data-файла."""
    text = lmp_path.read_text()
    masses_match = re.search(r"Masses\s*\n\s*\n(.*?)\n\s*\n", text, re.DOTALL)
    if not masses_match:
        raise ValueError(f"Не найдена секция Masses в {lmp_path}")
    type_to_mass: dict[int, float] = {}
    for line in masses_match.group(1).strip().splitlines():
        parts = line.split()
        type_to_mass[int(parts[0])] = float(parts[1])
    species = []
    for t in sorted(type_to_mass):
        m = type_to_mass[t]
        # ближайший элемент по атомной массе (с защитой от случайных совпадений)
        sym, diff = min(
            ((s, abs(ref - m)) for s, ref in ATOMIC_MASSES.items()),
            key=lambda x: x[1],
        )
        if diff > 0.5:
            raise ValueError(
                f"Не удалось сопоставить массу {m} с элементом (ближайший {sym} отстоит на {diff:.2f}); "
                "расширь ATOMIC_MASSES."
            )
        species.append(sym)
    return species


def select_rows(csv_path: Path) -> dict[str, dict]:
    """Группирует строки CSV по material, возвращает лучшую (kim > local) на материал."""
    by_material: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8") as f:
        filtered_lines = (line for line in f if not line.strip().startswith('#'))
        for row in csv.DictReader(filtered_lines):
            mat = row["material"]
            if row["status"] not in STATUS_PRIORITY:
                continue
            if mat not in by_material:
                by_material[mat] = row
                continue
            if STATUS_PRIORITY[row["status"]] < STATUS_PRIORITY[by_material[mat]["status"]]:
                by_material[mat] = row
    return by_material


def render(template: str, **kwargs) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    return out


def make_blocks(row: dict, species: list[str], material: str) -> tuple[str, str, str, list[str]]:
    """
    Возвращает (init_block, interactions_block, header_line, warnings).
    """
    species_str = " ".join(species)
    warnings: list[str] = []
    if row["status"] == "kim":
        kim_id = row["potential_ref"]
        init = f"kim init {kim_id} metal"
        interactions = f"kim interactions {species_str}"
        header = f"Material: {material}  | KIM: {kim_id}"
    elif row["status"] == "local":
        pair_style = row["pair_style"]
        file_rel = row["file"] or ""
        pot_path = SCRIPT_DIR / file_rel if file_rel else None
        if pot_path is None or not pot_path.exists():
            warnings.append(
                f"Файл потенциала не найден: {file_rel}. "
                "Скрипт будет сгенерирован как STUB — добавьте файл и перегенерируйте."
            )
            init = f"# TODO: файл потенциала {file_rel} отсутствует\npair_style       {pair_style}"
            interactions = f"# TODO: pair_coeff будет рабочим после добавления {file_rel}\npair_coeff      * * {file_rel} {species_str}"
        else:
            init = f"pair_style       {pair_style}"
            interactions = f"pair_coeff      * * {file_rel} {species_str}"
        header = f"Material: {material}  | Local: {row['potential_ref']} ({pair_style})"
    else:
        raise ValueError(f"Unsupported status: {row['status']}")
    return init, interactions, header, warnings


def main() -> None:
    INPUTS_DIR.mkdir(exist_ok=True)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    selected = select_rows(CSV_PATH)

    # Определяем список материалов для обработки
    if len(sys.argv) > 1:
        # Явно указаны имена материалов (как в CSV)
        requested = sys.argv[1:]
        # Проверяем, что все запрошенные материалы есть в selected
        missing = [m for m in requested if m not in selected]
        if missing:
            print(f"Ошибка: следующие материалы не найдены в CSV или имеют неподходящий статус: {', '.join(missing)}")
            sys.exit(1)
        materials_to_process = {m: selected[m] for m in requested}
    else:
        # Обрабатываем все доступные
        materials_to_process = selected

    written: list[str] = []
    skipped: list[tuple[str, str]] = []
    stubs: list[str] = []

    for material, row in sorted(materials_to_process.items()):
        safe = safe_material_name(material)
        lmp_file = f"{safe}.lmp"
        lmp_path = LAMMPS_INIT_DIR / lmp_file
        if not lmp_path.exists():
            skipped.append((material, f"нет lammps_init/{lmp_file}"))
            continue
        try:
            species = parse_species_from_lmp(lmp_path)
        except Exception as e:
            skipped.append((material, f"парсинг Masses не удался: {e}"))
            continue
        init, interactions, header, warns = make_blocks(row, species, material)
        rendered = render(
            template,
            HEADER_LINE=header,
            MATERIAL=material,
            LMP_FILE=lmp_file,
            INIT_BLOCK=init,
            INTERACTIONS_BLOCK=interactions,
        )
        out_path = INPUTS_DIR / f"in.{safe}"
        out_path.write_text(rendered, encoding="utf-8")
        written.append(material)
        if warns:
            stubs.append(material)
            for w in warns:
                print(f"  [STUB {material}] {w}")

    print(f"\nСгенерировано {len(written)} input-скриптов в {INPUTS_DIR.relative_to(SCRIPT_DIR)}/")
    print(f"  готовых к запуску: {len(written) - len(stubs)}")
    print(f"  stub'ов (нужен файл потенциала): {len(stubs)}")
    if stubs:
        print(f"  stubs: {', '.join(stubs)}")
    if skipped:
        print(f"\nПропущено {len(skipped)}:")
        for m, reason in skipped:
            print(f"  {m}: {reason}")


if __name__ == "__main__":
    main()

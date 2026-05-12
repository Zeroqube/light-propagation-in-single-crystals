"""
TODO: проверить анизотропность взятых кристаллов.
      посмотреть потенциалы для отдельных элементов.
"""
"""
mp_init.py
Скрипт для поиска mp-id двулучепреломляющих кристаллов из crystals_540nm.csv
Результат: crystals_540nm_with_ids.csv (исходный CSV с новым столбцом mp_id)
"""

import csv
from pathlib import Path
from mp_api.client import MPRester
from pymatgen.core import Structure

ROOT = Path(__file__).parent.parent

# === НАСТРОЙКИ ===
API_KEY = ""  # замените на реальный ключ
INPUT_CSV = ROOT / "crystals-search" / "crystals_540nm.csv"
OUTPUT_CSV = ROOT/ "md-simulation" / "crystals_540nm_mp.csv"

# === БЕЛЫЙ СПИСОК ИЗВЕСТНЫХ MP-ID (проверенные фазы) ===
KNOWN_IDS = {
    # "CaGdAlO4": "mp-1079296",   # чето нет данных на MP
    # "CaYAlO4": "mp-1227264",    # чето нет данных на MP
    "BaB2O4": None,      # β-BaB2O4, R3c
    # "BaB2O4": "mp-566932",      # β-BaB2O4, R3c
    "CsLiB6O10": "mp-1021352",  # CLBO, I-42d
    "LuAl3(BO3)4": "mp-504573", # R32
    "SiC": "mp-1204356",           # вручную, много атомов
    "CaCO3": "mp-3953",         # кальцит, R-3c
    "LaF3": "mp-2664",          # P-3c1 (тригональная)
    "LiCaAlF6": "mp-6134",     # P-31c (LiCAF)
    "MgF2": "mp-1249",          # P4₂/mnm (рутил)
    "YLiF4": "mp-3700",         # вручную
    "Pb5Ge3O11": None,          # в MP нет, искать в других базах
    "LiIO3": "mp-23352",        # P6₃ (нецентросимм.)
    "CaMoO4": "mp-19090",       # I4₁/a
    "PbMoO4": "mp-20418",       # I4₁/a
    "SrMoO4": "mp-22640",       # I4₁/a
    "AlN": "mp-661",            # P6₃mc
    "BN": "mp-604884",             # вручную
    "GaN": "mp-804",            # P6₃mc
    "LiNbO3": "mp-3731",        # R3c
    "Al2O3": "mp-1143",         # R-3c (корунд)
    "BeO": "mp-2542",           # P6₃mc
    "ScAlMgO4": "mp-1096925",   # R-3m
    "SiO2": "mp-7000",          # α-кварц, P3₂21 (левый/правый варианты)
    "TeO2": "mp-557",          # вручную, но хз
    "TiO2": "mp-2657",          # рутил, P4₂/mnm
    "ZnO": "mp-2133",           # P6₃mc
    "AlPO4": "mp-5331",         # вручную
    "KH2PO4": "mp-696752",       # вручную
    "NH4H2PO4": "mp-6978",      # ADP, I-42d
    "AgGaS2": "mp-5342",        # I-42d
    "CdS": "mp-672",            # P6₃mc (вюрцит)
    "CdGa2S4": None,            # вероятно, нет в MP
    "GaS": "mp-2507",           # вручную (хз)
    "LiTaO3": "mp-3666",        # R3c
    "BaTiO3": "mp-5986",         # вручную
    "PbTiO3": "mp-20459",       # P4mm
    "YVO4": "mp-19133",         # I4₁/amd (циркон)
    "CaWO4": "mp-18921",        # I4₁/a
}

def get_mp_id(formula, mpr):
    """Возвращает mp_id наиболее подходящей некубической фазы, или None."""
    try:
        docs = mpr.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "energy_above_hull", 
                    "symmetry"]
        )
    except Exception as e:
        print(f"  Ошибка поиска '{formula}': {e}")
        return None

    # Отбрасываем кубические (для одноосных кристаллов)
    non_cubic = [d for d in docs if d.symmetry and d.symmetry.crystal_system != "Cubic"]
    if not non_cubic:
        # Если все кубические, берём первую (вдруг это ошибка)
        non_cubic = docs

    if not non_cubic:
        return None

    # Сортируем по energy_above_hull, берём самую стабильную
    non_cubic.sort(key=lambda x: x.energy_above_hull)
    best = non_cubic[0]
    print(f"  Найдено {len(docs)} записей, выбрана {best.material_id} "
          f"({best.formula_pretty}), SG: {best.symmetry.symbol}, "
          f"E_hull={best.energy_above_hull:.3f} eV")
    return best.material_id

def main():
    with MPRester(API_KEY) as mpr:
        # Читаем исходный CSV
        with open(INPUT_CSV, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            fieldnames = reader.fieldnames

        # Добавляем колонку mp_id (если её ещё нет)
        if "mp_id" not in fieldnames:
            fieldnames.append("mp_id")

        for row in rows:
            name = row["book"]  # имя кристалла из столбца 'book'
            print(f"\nОбрабатываю: {name}")

            # 1. Проверяем белый список
            if name in KNOWN_IDS:
                mp_id = KNOWN_IDS[name]
                if mp_id is None:
                    row["mp_id"] = ""
                else:
                    row["mp_id"] = mp_id
                    print(f"  Использован ID из белого списка: {mp_id}")

                    if isinstance(mp_id, str) and mp_id:
                        struct = mpr.get_structure_by_material_id(mp_id)
                        struct.to(filename=ROOT / "md-simulation" / "mp_init" / f"{name}.cif", fmt="cif")
                        struct.to(filename=ROOT / "md-simulation" / "mp_init" / f"{name}.vasp", fmt="poscar")
                    else:
                        print(f"  Пропущено (mp_id не найден)")
                continue

            # 2. Автопоиск по формуле (убираем скобки, пробелы)
            formula = name.replace(" ", "").replace("(", "").replace(")", "")
            print(f"  Поиск по формуле: {formula}")
            mp_id = get_mp_id(formula, mpr)
            row["mp_id"] = mp_id if mp_id else ""

            if isinstance(mp_id, str) and mp_id:
                struct = mpr.get_structure_by_material_id(mp_id)
                struct.to(filename=ROOT / "md-simulation" / "mp_init" / f"{name}.cif", fmt="cif")
                struct.to(filename=ROOT / "md-simulation" / "mp_init" / f"{name}.vasp", fmt="poscar")
            else:
                print(f"  Пропущено (mp_id не найден)")

        # Сохраняем результат
        with open(OUTPUT_CSV, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nГотово! Результат записан в {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

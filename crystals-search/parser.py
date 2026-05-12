"""
pip install refractiveindex

модуль refractiveindex чуть криво работает; я весь код из ".refractiveindex" который вызывается в 
RefractiveIndexMaterial вставил напрямую в __init__.py файл; только таким образом заработало.

В общем, код ищет данные в директориях соединений с суффиксами связанными с "o", "e" и считает по дисперсионной формуле
https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf 
значения коэффициентов двулучепреломления для фиксированной длины волны, после чего сохраняет в файле .csv
"""


import sys
import csv
import re
import warnings
import argparse
from pathlib import Path
import numpy as np
import yaml
from refractiveindex import RefractiveIndexMaterial, _load_catalog, NoExtinctionCoefficient


def get_suffix_from_page(page_name):
    """Возвращает 'o' или 'e', если имя страницы оканчивается на -o, _o, -e, _e (до .yml)."""
    base = page_name
    if base.endswith('.yml'):
        base = base[:-4]
    if base.endswith('-o') or base.endswith('_o'):
        return 'o'
    if base.endswith('-e') or base.endswith('_e'):
        return 'e'
    return None


def safe_n_and_k(mat, wavelength_um):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            n = mat.get_refractive_index(wavelength_um, unit='um')
            if np.isnan(n) or np.isinf(n) or n < 1.2:
                return None, None
            try:
                k = mat.get_extinction_coefficient(wavelength_um, unit='um')
                if np.isnan(k) or np.isinf(k):
                    k = 0.0
            except NoExtinctionCoefficient:
                k = 0.0
            return n, k
        except:
            return None, None


def collect_uniaxial_pairs(db_path, wavelength_um=0.63, min_diff=0.01, max_k=0.01):
    """Поиск пар o/e с достаточным двулучепреломлением."""
    catalog = _load_catalog(db_path)
    records = []  # (shelf, book, suffix, n, page, k)

    for (shelf, book, page), yaml_path in catalog.items():
        if shelf != 'main':      # явно только главная полка (можно добавить 'organic')
            continue

        suffix = get_suffix_from_page(page)
        if suffix is None:
            continue

        try:
            mat = RefractiveIndexMaterial(shelf, book, page, db_path=db_path,
                                          auto_download=False)
        except Exception:
            continue

        wl_range = mat.get_wl_range(unit='um')
        if wl_range is None:
            continue
        wl_min, wl_max = wl_range
        if not (wl_min <= wavelength_um <= wl_max):
            continue

        n, k = safe_n_and_k(mat, wavelength_um)
        if n is None or k > max_k:
            continue

        records.append((shelf, book, suffix, n, page, k))

    # Группируем по (shelf, book)
    from collections import defaultdict
    books = defaultdict(lambda: {'o': [], 'e': []})
    for (shelf, book, suffix, n, page, k) in records:
        books[(shelf, book)][suffix].append((n, page, k))

    candidates = {}
    for (shelf, book), dirs in books.items():
        o_list = dirs['o']
        e_list = dirs['e']
        if not o_list or not e_list:
            continue
        best_o = min(o_list, key=lambda x: x[2])
        best_e = min(e_list, key=lambda x: x[2])
        n_o, page_o, k_o = best_o
        n_e, page_e, k_e = best_e
        if abs(n_o - n_e) >= min_diff:
            candidates[(shelf, book)] = (n_o, n_e, page_o, page_e)

    return candidates


def compute_material_refractive_indices(db_path, material_name, wavelength_um):
    """
    Для заданного имени материала (book) находит все страницы в полке 'main',
    вычисляет n и k на заданной длине волны.
    Если ни одна страница не подходит, откатывается к поиску на серединах диапазонов.
    Возвращает список кортежей (shelf, book, page, n, k, actual_wavelength_um, fallback_flag)
    """
    catalog = _load_catalog(db_path)
    results = []
    # Сначала собираем все страницы материала, не фильтруя по длине волны
    all_pages = []
    for (shelf, book, page), yaml_path in catalog.items():
        if shelf != 'main':
            continue
        if book.lower() != material_name.lower():
            continue
        all_pages.append((shelf, book, page))

    if not all_pages:
        return results  # материал не найден

    # Попытка с точной длиной волны
    exact_fit = False
    for shelf, book, page in all_pages:
        try:
            mat = RefractiveIndexMaterial(shelf, book, page, db_path=db_path,
                                          auto_download=False)
        except Exception:
            continue
        wl_range = mat.get_wl_range(unit='um')
        if wl_range is None:
            continue
        wl_min, wl_max = wl_range
        if not (wl_min <= wavelength_um <= wl_max):
            continue
        n, k = safe_n_and_k(mat, wavelength_um)
        if n is not None:
            results.append((shelf, book, page, n, k, wavelength_um, False))
            exact_fit = True

    if exact_fit:
        return results

    # Если ничего не найдено – fallback: для каждой страницы берём середину диапазона
    print(f"Внимание: для материала '{material_name}' нет данных на длине волны {wavelength_um} мкм. "
          "Будут использованы значения в серединах рабочих диапазонов страниц.", file=sys.stderr)
    for shelf, book, page in all_pages:
        try:
            mat = RefractiveIndexMaterial(shelf, book, page, db_path=db_path,
                                          auto_download=False)
        except Exception:
            continue
        wl_range = mat.get_wl_range(unit='um')
        if wl_range is None:
            # Если диапазон неизвестен, пробуем заданную длину (вдруг получится)
            fallback_wl = wavelength_um
        else:
            fallback_wl = (wl_range[0] + wl_range[1]) / 2.0
        n, k = safe_n_and_k(mat, fallback_wl)
        if n is not None:
            results.append((shelf, book, page, n, k, fallback_wl, True))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Поиск показателей преломления в локальной базе refractiveindex.info')
    parser.add_argument('--material', type=str, default=None,
                        help='Название материала (book) для точечного расчёта n,k')
    parser.add_argument('--wavelength', type=float, default=0.54,
                        help='Длина волны в мкм (по умолчанию 0.54)')
    parser.add_argument('--min-diff', type=float, default=0.001,
                        help='Минимальная разница n_o - n_e для двулучепреломления')
    parser.add_argument('--max-k', type=float, default=0.01,
                        help='Максимально допустимый коэффициент экстинкции')
    args = parser.parse_args()

    db_path = Path(__file__).parent / "database"
    wavelength_um = args.wavelength

    if args.material:
        # Режим точечного поиска материала
        material = args.material
        entries = compute_material_refractive_indices(db_path, material, wavelength_um)
        if not entries:
            print(f"Материал '{material}' не найден в полке main.")
        else:
            print(f"Найдено страниц для '{material}': {len(entries)}")
            for shelf, book, page, n, k, used_wl, fallback in entries:
                note = " (fallback)" if fallback else ""
                print(f"  {page}: n={n:.6f}, k={k:.6f} @ {used_wl:.4f} мкм{note}")
            # Сохранение в CSV
            safe_name = re.sub(r'[\\/*?:"<>|]', "_", material)
            output = f"crystals-search/{safe_name}_{wavelength_um*1000:.0f}nm.csv"
            with open(output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['shelf', 'book', 'page', 'n', 'k', 'wavelength_um', 'fallback'])
                for shelf, book, page, n, k, used_wl, fallback in entries:
                    writer.writerow([shelf, book, page, n, k, used_wl, fallback])
            print(f"Результат сохранён в {output}")
    else:
        # Режим поиска двулучепреломляющих пар (исходное поведение)
        pairs = collect_uniaxial_pairs(db_path, wavelength_um=wavelength_um,
                                       min_diff=args.min_diff, max_k=args.max_k)
        print(f"Отфильтровано (только o/e суффиксы, без поглощения): {len(pairs)}")
        for (shelf, book), (n1, n2, p1, p2) in pairs.items():
            print(f"{shelf}/{book}: {n1:.6f} ({p1}), {n2:.6f} ({p2})")
        output = f"crystals-search/crystals_{wavelength_um*1000:.0f}nm.csv"
        with open(output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['shelf', 'book', 'n_o', 'n_e', 'page_o', 'page_e', 'wavelength_um'])
            for (shelf, book), (n_o, n_e, page_o, page_e) in pairs.items():
                writer.writerow([shelf, book, n_o, n_e, page_o, page_e, wavelength_um])
        print(f"\nРезультаты сохранены в {output}")

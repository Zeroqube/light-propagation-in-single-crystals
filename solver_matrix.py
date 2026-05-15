'''
Todo: подумать над оптимизацией: tau1, нет смысла считать от -N, N, посчитать одну четверть и умножить на 4?
Написать tau2, сложение матрицы, взятие обратной
'''


import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.transform import Rotation

### Дальше часть кода про поворот кристалла, взятая с дипсика

def get_lattice_dimensions(structure: Structure):
    """
    Возвращает размеры ячейки по осям X и Y (поперечные направления).
    
    Returns:
        Lx, Ly: размеры в Å
    """
    lattice = structure.lattice.matrix
    Lx = np.linalg.norm(lattice[0])  # вектор a
    Ly = np.linalg.norm(lattice[1])  # вектор b
    Lz = np.linalg.norm(lattice[2])  # вектор c
    return Lx, Ly, Lz

def get_optical_axis_direction(structure: Structure):
    """
    Определяет направление оптической оси в декартовых координатах.
    
    Для одноосных кристаллов (тригональные, гексагональные, тетрагональные)
    оптическая ось совпадает с осью c (третий вектор решётки).
    
    Для кубических — возвращает None (изотропный).
    Для двуосных — нужно задать вручную.
    
    Возвращает:
        numpy.ndarray: единичный вектор оптической оси (в декартовых координатах)
        str: тип ('uniaxial', 'cubic', 'biaxial')
    """
    # Приводим к стандартной ячейке
    sga = SpacegroupAnalyzer(structure)
    crystal_system = sga.get_crystal_system()
    conventional = sga.get_conventional_standard_structure()
    
    print(f"Сингония: {crystal_system}")
    
    if crystal_system in ['trigonal', 'hexagonal', 'tetragonal']:
        # Одноосные кристаллы: оптическая ось = ось c
        c_vector = conventional.lattice.matrix[2]
        optical_axis = c_vector / np.linalg.norm(c_vector)
        return optical_axis, 'uniaxial'
    
    elif crystal_system == 'cubic':
        # Изотропный — оптической оси нет
        return None, 'cubic'
    
    else:
        # Двуосные или другие: нужно задавать вручную
        return None, 'biaxial'


def rotate_structure_optical_axis_to_x(structure: Structure, manual_axis: np.ndarray = None):
    """
    Поворачивает кристалл так, чтобы оптическая ось стала вдоль оси X.
    
    Параметры:
        structure: исходная структура
        manual_axis: вектор оптической оси (для двуосных кристаллов)
        
    Возвращает:
        rotated_structure: повёрнутая структура
        optical_direction: итоговое направление оптической оси (должно быть (1,0,0))
    """
    # Определяем оптическую ось
    optical_axis, crystal_type = get_optical_axis_direction(structure)
    
    if crystal_type == 'cubic':
        print("Кубический кристалл (изотропный) — поворот не требуется")
        return structure, np.array([1.0, 0.0, 0.0])
    
    if crystal_type == 'biaxial':
        if manual_axis is None:
            raise ValueError(f"Двуосный кристалл. Задайте оптическую ось вручную через manual_axis")
        optical_axis = manual_axis / np.linalg.norm(manual_axis)
        print(f"Использую ручное направление: {optical_axis}")
    
    # Целевое направление — ось X
    target_dir = np.array([1.0, 0.0, 0.0])
    
    # Находим матрицу поворота
    v = np.cross(optical_axis, target_dir)
    s = np.linalg.norm(v)
    c = np.dot(optical_axis, target_dir)
    
    if s < 1e-10:
        # Уже совпадает
        rotation_matrix = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    
    # Поворачиваем структуру
    rotated = structure.copy()
    rotated.rotate_sites(rotation_matrix, to_unit_cell=False)
    
    # Также поворачиваем решётку
    new_lattice_matrix = np.dot(rotation_matrix, structure.lattice.matrix.T).T
    rotated.lattice.matrix = new_lattice_matrix
    
    return rotated, np.array([1.0, 0.0, 0.0])


def replicate_along_z(structure: Structure, N: int):
    """
    Размножает структуру вдоль оси Z.
    
    Параметры:
        structure: исходная структура
        N: количество ячеек вдоль Z
        
    Возвращает:
        x, y, z: массивы координат всех атомов
        types: массив типов атомов
    """
    # Получаем вектор вдоль Z (третий вектор решётки)
    z_vector = structure.lattice.matrix[2]
    z_thickness = np.linalg.norm(z_vector)
    
    # Забираем координаты исходных атомов
    x_list = []
    y_list = []
    z_list = []
    types_list = []
    
    for site in structure:
        x_list.append(site.coords[0])
        y_list.append(site.coords[1])
        z_list.append(site.coords[2])
        types_list.append(site.species_string)
    
    x_0 = np.array(x_list)
    y_0 = np.array(y_list)
    z_0 = np.array(z_list)
    types_0 = np.array(types_list)
    
    # Размножаем
    x_all = []
    y_all = []
    z_all = []
    types_all = []
    
    for i in range(N):
        shift_z = i * z_thickness
        x_all.append(x_0)
        y_all.append(y_0)
        z_all.append(z_0 + shift_z)
        types_all.append(types_0)
    
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    z_all = np.concatenate(z_all)
    types_all = np.concatenate(types_all)
    
    return x_all, y_all, z_all, types_all


### дальше моя реализация физики
omega = 1e15
c = 3e8 ### СИ или СГС?
cif_path = '/home/ubun/projects/light-propagation-in-single-crystals/md-simulation/output/unit_cells/CaCO3.cif'
struct = Structure.from_file(cif_path)

params = {
        'Ca' : {'m' : 40, 'k' : 1.0}, 
        'O' : {'m' : 16, 'k' : 1.0},
}       

N_perp = 0 # количество слоёв вверх и столько же вниз

N = 1 # кол-во ячеек вдоль направления распространения

def tau1(params, struct):
    struct, _ = rotate_structure_optical_axis_to_x(struct)
    Lx, Ly, Lz = get_lattice_dimensions(struct)
    x_all, y_all, z_all, types_all = replicate_along_z(struct, N)
    S = len(types_all) / N
    matr = np.zeros((3 * N * S, 3 * N * S), dtype = 'complex')
    for i in range(N * S):
        x0 = x_all[i]
        y0 = y_all[i]
        z0 = z_all[i]
        row0 = matr[3 * i]
        row1 = matr[3 * i + 1]
        row2 = matr[3 * i + 2]
        for j in range(N * S):
                x = x_all[j]
                y = y_all[j]
                z = z_all[j]
                sum = np.zeros((3, 3), dtype = 'complex')
                for l in range(- N_perp, N_perp + 1):
                        for n in range(-N_perp, N_perp + 1):
                                if i == j and l == 0 and n == 0:
                                        continue
                                x_cur = x + Lx * l
                                y_cur = y + Ly * n
                                z_cur = z
                                lx = x0 - x_cur
                                ly = y0 - y_cur
                                lz = z0 - z_cur
                                r = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
                                

                                # E_x = (3 * (dx * lx + dy * ly + dz * lz) lx / r ** 5 - lx / r ** 3) * np.exp(- 1j * omega * r / c)
                                # E_y = (3 * (dx * lx + dy * ly + dz * lz) ly / r ** 5 - ly / r ** 3) * np.exp(- 1j * omega * r / c)
                                # E_z = (3 * (dx * lx + dy * ly + dz * lz) lz/ r ** 5 - lz / r ** 3) * np.exp(- 1j * omega * r / c)

                                sum[0][0] += (3 * lx**2 / r**5 - 1 / r**3) * np.exp(- 1j * omega * r / c)
                                sum[0][1] += (3 * ly * lx / r**5) * np.exp(- 1j * omega * r / c)
                                sum[0][2] += (3 * lz * lx / r**5) * np.exp(- 1j * omega * r / c)

                                sum[1][0] += (3 * lx * ly / r**5) * np.exp(- 1j * omega * r / c)
                                sum[1][1] += (3 * ly**2 / r**5 - 1 / r**3) * np.exp(- 1j * omega * r / c)
                                sum[1][2] += (3 * lz * ly / r**5) * np.exp(- 1j * omega * r / c)

                                sum[2][0] += (3 * lz * lx / r**5) * np.exp(- 1j * omega * r / c)
                                sum[2][1] += (3 * ly * lz / r**5) * np.exp(- 1j * omega * r / c)
                                sum[2][2] += (3 * lz**2 / r**5 - 1 / r**3) * np.exp(- 1j * omega * r / c)

                for p in range(3):
                        row0[3 * j + p] = sum[0][p]
                        row1[3 * j + p] = sum[1][p]
                        row2[3 * j + p] = sum[2][p]
        return matr
                





                    






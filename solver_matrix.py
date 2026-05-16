'''
Todo: подумать над оптимизацией: tau1, нет смысла считать от -N, N, посчитать одну четверть и умножить на 4?

Замечание: матрицы tau1, tau2, tau3 не зависят от параметров, которые оптимизируем, их достаточно посчитать один раз.
'''


import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time

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


def rotate_structure_optical_axis_to_x(
    structure: Structure,
    manual_axis: np.ndarray = None
):

    optical_axis, crystal_type = get_optical_axis_direction(structure)

    if crystal_type == 'cubic':
        return structure, np.array([1.0, 0.0, 0.0])

    if crystal_type == 'biaxial':
        if manual_axis is None:
            raise ValueError("Для двуосного кристалла задайте manual_axis")

        optical_axis = manual_axis / np.linalg.norm(manual_axis)

    target_dir = np.array([1.0, 0.0, 0.0])

    optical_axis = optical_axis / np.linalg.norm(optical_axis)

    v = np.cross(optical_axis, target_dir)
    s = np.linalg.norm(v)
    c = np.dot(optical_axis, target_dir)

    if s < 1e-10:

        if c > 0:
            rotation_matrix = np.eye(3)

        else:
            perp = np.array([0.0, 1.0, 0.0])

            rot_axis = np.cross(optical_axis, perp)
            rot_axis = rot_axis / np.linalg.norm(rot_axis)

            rotation_matrix = Rotation.from_rotvec(
                np.pi * rot_axis
            ).as_matrix()

    else:
        rot_axis = v / s

        angle = np.arccos(np.clip(c, -1.0, 1.0))

        rotation_matrix = Rotation.from_rotvec(
            angle * rot_axis
        ).as_matrix()

    rotated = structure.copy()

    # Вращаем атомы
    for idx, site in enumerate(rotated.sites):

        new_coords = rotation_matrix @ site.coords

        rotated.replace(
            idx,
            site.species,
            coords=new_coords,
            coords_are_cartesian=True
        )

    # Вращаем решетку
    new_lattice_matrix = (
        rotation_matrix @ structure.lattice.matrix.T
    ).T

    from pymatgen.core import Lattice

    rotated.lattice = Lattice(new_lattice_matrix)

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
        x_list.append(site.coords[0] * 1e-8)
        y_list.append(site.coords[1] * 1e-8)
        z_list.append(site.coords[2] * 1e-8)
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
omega = 3e15
c = 2.998e10 
e = 4.803e-10

cif_path = '/home/ubun/projects/light-propagation-in-single-crystals/md-simulation/output/unit_cells/CaCO3.cif'
struct = Structure.from_file(cif_path)

params = {
    'C'  : {'m': 1.6e-26,  'k': 4.7e6},
    'O'  : {'m': 4.3e-26,  'k': 1.8e7},
    'Mg' : {'m': 2.3e-26,  'k': 3.1e6},
    'S'  : {'m': 8.3e-26,  'k': 2.0e7},
    'Ca' : {'m': 4.7e-26,  'k': 4.1e6},
    'Zn' : {'m': 1.8e-25,  'k': 3.6e7},
    'Cd' : {'m': 4.0e-25,  'k': 7.4e7},
}      

N_perp = 10 # количество слоёв вверх и столько же вниз

N = 10 # кол-во ячеек вдоль направления распространения

def tau1(Lx, Ly, x_all, y_all, z_all, L):
        matr = np.zeros((3 * L, 3 * L), dtype = 'complex')
        for i in range(L):
                x0 = x_all[i]
                y0 = y_all[i]
                z0 = z_all[i]
                row0 = matr[3 * i]
                row1 = matr[3 * i + 1]
                row2 = matr[3 * i + 2]
                for j in range(L):
                        x = x_all[j]
                        y = y_all[j]
                        z = z_all[j]
                        sum = np.zeros((3, 3), dtype = 'complex')
                        for nx in range(- N_perp, N_perp + 1):
                                for ny in range(-N_perp, N_perp + 1):
                                        if i == j and nx == 0 and ny == 0:
                                                continue
                                        x_cur = x + Lx * nx
                                        y_cur = y + Ly * ny
                                        z_cur = z
                                        lx = x0 - x_cur
                                        ly = y0 - y_cur
                                        lz = z0 - z_cur
                                        r = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
                                        

                                        # E_x = (3 * (dx * lx + dy * ly + dz * lz) lx / r ** 5 - lx / r ** 3) * np.exp(- 1j * omega * r / c)
                                        # E_y = (3 * (dx * lx + dy * ly + dz * lz) ly / r ** 5 - ly / r ** 3) * np.exp(- 1j * omega * r / c)
                                        # E_z = (3 * (dx * lx + dy * ly + dz * lz) lz/ r ** 5 - lz / r ** 3) * np.exp(- 1j * omega * r / c)

                                        phase = np.exp(-1j * omega * r / c)

                                        sum[0][0] += (3 * lx**2 / r**5 - 1 / r**3) * phase
                                        sum[0][1] += (3 * ly * lx / r**5) * phase
                                        sum[0][2] += (3 * lz * lx / r**5) * phase

                                        sum[1][0] += (3 * lx * ly / r**5) * phase
                                        sum[1][1] += (3 * ly**2 / r**5 - 1 / r**3) * phase
                                        sum[1][2] += (3 * lz * ly / r**5) * phase

                                        sum[2][0] += (3 * lz * lx / r**5) * phase
                                        sum[2][1] += (3 * ly * lz / r**5) * phase
                                        sum[2][2] += (3 * lz**2 / r**5 - 1 / r**3) * phase

                        for p in range(3):
                                row0[3 * j + p] = sum[0][p]
                                row1[3 * j + p] = sum[1][p]
                                row2[3 * j + p] = sum[2][p]
        return matr
                
def tau2(Lx, Ly, x_all, y_all, z_all, L):
        matr = np.zeros((3 * L, 3 * L), dtype = 'complex')
        for i in range(L):
                x0 = x_all[i]
                y0 = y_all[i]
                z0 = z_all[i]
                row0 = matr[3 * i]
                row1 = matr[3 * i + 1]
                row2 = matr[3 * i + 2]
                for j in range(L):
                        x = x_all[j]
                        y = y_all[j]
                        z = z_all[j]
                        sum = np.zeros((3, 3), dtype = 'complex')
                        for nx in range(- N_perp, N_perp + 1):
                                for ny in range(-N_perp, N_perp + 1):
                                        if i == j and nx == 0 and ny == 0:
                                                continue
                                        x_cur = x + Lx * nx
                                        y_cur = y + Ly * ny
                                        z_cur = z
                                        lx = x0 - x_cur
                                        ly = y0 - y_cur
                                        lz = z0 - z_cur
                                        r = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
                                        

                                        # E_x = (3 * (dx * lx + dy * ly + dz * lz) lx / r ** 4 - lx / r ** 2) * np.exp(- 1j * omega * r / c) / c 
                                        # E_y = (3 * (dx * lx + dy * ly + dz * lz) ly / r ** 4 - ly / r ** 2) * np.exp(- 1j * omega * r / c) / c
                                        # E_z = (3 * (dx * lx + dy * ly + dz * lz) lz/ r ** 4 - lz / r ** 2) * np.exp(- 1j * omega * r / c) / c

                                        phase = np.exp(-1j * omega * r / c)

                                        sum[0][0] += (3 * lx**2 / r**4 - 1 / r**2) * phase / c
                                        sum[0][1] += (3 * ly * lx / r**4) * phase / c
                                        sum[0][2] += (3 * lz * lx / r**4) * phase / c

                                        sum[1][0] += (3 * lx * ly / r**4) * phase / c
                                        sum[1][1] += (3 * ly**2 / r**4 - 1 / r**2) * phase / c
                                        sum[1][2] += (3 * lz * ly / r**4) * phase / c

                                        sum[2][0] += (3 * lz * lx / r**4) * phase / c
                                        sum[2][1] += (3 * ly * lz / r**4) * phase / c
                                        sum[2][2] += (3 * lz**2 / r**4 - 1 / r**2) * phase / c

                        for p in range(3):
                                row0[3 * j + p] = sum[0][p]
                                row1[3 * j + p] = sum[1][p]
                                row2[3 * j + p] = sum[2][p]
        return matr


def tau3(Lx, Ly, x_all, y_all, z_all, L):
        matr = np.zeros((3 * L, 3 * L), dtype = 'complex')
        for i in range(L):
                x0 = x_all[i]
                y0 = y_all[i]
                z0 = z_all[i]
                row0 = matr[3 * i]
                row1 = matr[3 * i + 1]
                row2 = matr[3 * i + 2]
                for j in range(L):
                        x = x_all[j]
                        y = y_all[j]
                        z = z_all[j]
                        sum = np.zeros((3, 3), dtype = 'complex')
                        for nx in range(- N_perp, N_perp + 1):
                                for ny in range(-N_perp, N_perp + 1):
                                        if i == j and nx == 0 and ny == 0:
                                                continue
                                        x_cur = x + Lx * nx
                                        y_cur = y + Ly * ny
                                        z_cur = z
                                        lx = x0 - x_cur
                                        ly = y0 - y_cur
                                        lz = z0 - z_cur
                                        r = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
                                        

                                        # E_x = ((dx * lx + dy * ly + dz * lz) lx / r ** 3 - lx / r) * np.exp(- 1j * omega * r / c) / c**2
                                        # E_y = ((dx * lx + dy * ly + dz * lz) ly / r ** 3 - ly / r) * np.exp(- 1j * omega * r / c) / c**2
                                        # E_z = ((dx * lx + dy * ly + dz * lz) lz/ r ** 3 - lz / r) * np.exp(- 1j * omega * r / c) / c**2

                                        phase = np.exp(-1j * omega * r / c)

                                        sum[0][0] += (lx**2 / r**3 - 1 / r) * phase / c**2
                                        sum[0][1] += (ly * lx / r**3) * phase / c**2
                                        sum[0][2] += (lz * lx / r**3) * phase / c**2

                                        sum[1][0] += (lx * ly / r**3) * phase / c**2
                                        sum[1][1] += (ly**2 / r**3 - 1 / r) * phase / c**2
                                        sum[1][2] += (lz * ly / r**3) * phase / c**2

                                        sum[2][0] += (lz * lx / r**3) * phase / c**2
                                        sum[2][1] += (ly * lz / r**3) * phase / c**2
                                        sum[2][2] += (lz**2 / r**3 - 1 / r) * phase / c**2

                        for p in range(3):
                                row0[3 * j + p] = sum[0][p]
                                row1[3 * j + p] = sum[1][p]
                                row2[3 * j + p] = sum[2][p]
        return matr


def M(params,types_all, L):
        matr = np.eye(3 * L)
        for i in range(L):
                type = types_all[i]
                m = params[type]['m']
                for j in range(3):
                        matr[3*i + j] *= m / e **2
        return matr

def K(params, types_all, L):
        L = len(types_all)
        matr = np.eye(3 * L)
        for i in range(L):
                type = types_all[i]
                k = params[type]['k']
                for j in range(3):
                        matr[3 * i + j] *= k / e **2
        return matr

def q(struct, params, theta):
        print("Start...")
        struct, _ = rotate_structure_optical_axis_to_x(struct)
        Lx, Ly, Lz = get_lattice_dimensions(struct)
        x_all, y_all, z_all, types_all = replicate_along_z(struct, N)
        L = len(types_all)
        ### реплицирование, потом передавать уже нормальные ячейки
        tau_1 = tau1(Lx, Ly, x_all, y_all, z_all, L)
        tau_2 = tau2(Lx, Ly, x_all, y_all, z_all, L)
        tau_3 = tau3(Lx, Ly, x_all, y_all, z_all, L)
        M_matr = M(params, types_all, L)
        K_matr = K(params, types_all, L)
        A = -omega**2 * M_matr + K_matr - tau_1 -1j * omega * tau_2 + omega **2 * tau_3
        A_inv = np.linalg.inv(A)
        b = np.zeros(3 * L, dtype = 'complex')
        theta_rad = theta / 180 * np.pi
        for i in range(L):
              z = z_all[i]
              phase = np.exp(-1j * omega * z / c)
              b[3 * i] = np.cos(theta_rad) * phase
              b[3 * i + 1] = np.sin(theta_rad) * phase
              b[3 * i + 2] = 0
        q = A_inv @ b
        return q

def q_mod(struct, params, theta):
        print("Это q_mod")
        struct, _ = rotate_structure_optical_axis_to_x(struct)
        Lx, Ly, Lz = get_lattice_dimensions(struct)
        x_all, y_all, z_all, types_all = replicate_along_z(struct, N)
        L = len(types_all)
        ### реплицирование, потом передавать уже нормальные ячейки
        tau_1 = tau1(Lx, Ly, x_all, y_all, z_all, L)
        tau_2 = tau2(Lx, Ly, x_all, y_all, z_all, L)
        tau_3 = tau3(Lx, Ly, x_all, y_all, z_all, L)
        M_matr = M(params, types_all, L)
        K_matr = K(params, types_all, L)
        A = -omega**2 * M_matr + K_matr - tau_1 -1j * omega * tau_2 + omega **2 * tau_3
        b = np.zeros(3 * L, dtype = 'complex')
        theta_rad = theta / 180 * np.pi
        for i in range(L):
              z = z_all[i]
              phase = np.exp(-1j * omega * z / c)
              b[3 * i] = np.cos(theta_rad) * phase
              b[3 * i + 1] = np.sin(theta_rad) * phase
              b[3 * i + 2] = 0
        q = np.linalg.solve(A, b)
        return q

def phase_analysis(q_vec, cif_path, idx = 0):
        struct = Structure.from_file(cif_path)
        S = len(struct)
        q_atoms = q_vec.reshape(-1, 3)
        q_components = q_atoms.reshape(S, N, 3)
        phases = np.angle(q_components[idx])
        return phases

t1 = time.time()
q_ans = q_mod(struct, params, 0)
t2 = time.time()
print(f"Time: {t2 - t1}")
phases = phase_analysis(q_ans, cif_path)
# print(phases[:,0])
N_np = np.arange(N)
plt.plot(N_np, phases[:,0], label = 'x')
plt.errorbar(N_np, phases[:,0], fmt = '.')
# plt.plot(N_np, phases[:,1], label = 'y')
# plt.plot(N_np, phases[:,2], label = 'z')
plt.legend()
plt.grid()
plt.show()


                

                    






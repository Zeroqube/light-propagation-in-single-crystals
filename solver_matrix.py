'''
Todo: подумать над оптимизацией: tau1, нет смысла считать от -N, N, посчитать одну четверть и умножить на 4?
Плохо сдвигаю ячейку!

Замечание: матрицы tau1, tau2, tau3 не зависят от параметров, которые оптимизируем, их достаточно посчитать один раз.
'''

import numba
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import time

### Дальше часть кода про поворот кристалла, взятая с дипсика

ORTHO_MATRIX = [
    [1, 0, 0],
    [1, 2, 0],
    [0, 0, 1]
]

NEEDS_ORTHO = {"hexagonal", "trigonal"}


def get_lattice_vectors(structure):
    """
    Возвращает векторы решётки в сантиметрах (СГС).
    Матрица в pymatgen имеет форму (3, 3), где строки — это векторы a, b, c.
    """
    lattice = structure.lattice.matrix * 1e-8  # Из Ангстремов в см
    return lattice[0], lattice[1], lattice[2]

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
        return structure.copy(), np.array([1.0, 0.0, 0.0])

    if crystal_type == 'biaxial':
        if manual_axis is None:
            raise ValueError("Для двуосного кристалла задайте manual_axis")

        optical_axis = manual_axis / np.linalg.norm(manual_axis)

    optical_axis = optical_axis / np.linalg.norm(optical_axis)

    target = np.array([1.0, 0.0, 0.0])

    cross = np.cross(optical_axis, target)
    norm = np.linalg.norm(cross)

    if norm < 1e-12:

        if np.dot(optical_axis, target) > 0:
            rot_matrix = np.eye(3)
        else:
            rot_matrix = Rotation.from_rotvec(
                np.pi * np.array([0,1,0])
            ).as_matrix()

    else:
        axis = cross / norm
        angle = np.arccos(np.clip(np.dot(optical_axis, target), -1, 1))

        rot_matrix = Rotation.from_rotvec(
            angle * axis
        ).as_matrix()

    new_lattice = rot_matrix @ structure.lattice.matrix.T
    new_lattice = new_lattice.T

    new_coords = []

    for site in structure:
        new_coords.append(rot_matrix @ site.coords)

    rotated = Structure(
        lattice=Lattice(new_lattice),
        species=[site.species for site in structure],
        coords=new_coords,
        coords_are_cartesian=True
    )

    return rotated, target




def replicate_along_z(structure: Structure, N: int):
    """
    Правильная репликация ячейки вдоль вектора трансляции решётки.
    Поскольку волна идет вдоль z, предполагается, что после всех поворотов 
    вектор c_vec направлен (или соосен) с направлением трансляции кристалла.
    """
    # Получаем базовые декартовы координаты атомов в см
    coords0 = structure.cart_coords * 1e-8
    types0 = np.array([site.species_string for site in structure])
    
    # Получаем актуальный вектор трансляции c_vec (в см) из повернутой структуры
    _, _, c_vec = get_lattice_vectors(structure)
    
    x_all, y_all, z_all, types_all = [], [], [], []
    
    # Репликация происходит строго сдвигом на вектор решётки n * c_vec
    for n in range(N):
        shift = n * c_vec
        
        x_all.append(coords0[:, 0] + shift[0])
        y_all.append(coords0[:, 1] + shift[1])
        z_all.append(coords0[:, 2] + shift[2])
        types_all.append(types0)
        
    return (
        np.concatenate(x_all),
        np.concatenate(y_all),
        np.concatenate(z_all),
        np.concatenate(types_all)
    )


### дальше моя реализация физики
# omega = 3e15
omega = 9e17
c = 2.998e10 
e = 4.803e-10



params = {
    'C'  : {'m': 1.6e-26,  'k': 4.7e6},
    'O'  : {'m': 4.3e-26,  'k': 1.8e7},
    'Mg' : {'m': 3e-26,  'k': 3.1e7}, # 'm': 2.3e-26
    'S'  : {'m': 3e-26,  'k': 2.0e7}, # 'm': 8.3e-26
    'Ca' : {'m': 4.7e-26,  'k': 4.1e6},
    'Zn' : {'m': 1.8e-25,  'k': 3.6e7},
    'Cd' : {'m': 4.0e-25,  'k': 7.4e7},
}      

N_perp = 20 # количество слоёв вверх и столько же вниз

N = 1000 # кол-во ячеек вдоль направления распространения

@numba.jit
def tau1(a_vec, b_vec, x_all, y_all, z_all, L, N, N_perp):
        matr = np.zeros((3 * L, 3 * L), dtype = 'complex')
        for i in range(L):
                print('tau1:', i)
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
                                        shift = nx * a_vec + ny * b_vec
                                        x_cur = x + shift[0]
                                        y_cur = y + shift[1]
                                        z_cur = z + shift[2]
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
                
@numba.jit
def tau2(a_vec, b_vec, x_all, y_all, z_all, L, N, N_perp):
        matr = np.zeros((3 * L, 3 * L), dtype = 'complex')
        for i in range(L):
                print('tau2:', i)
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
                                        shift = nx * a_vec + ny * b_vec
                                        x_cur = x + shift[0]
                                        y_cur = y + shift[1]
                                        z_cur = z + shift[2]
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


@numba.jit
def tau3(a_vec, b_vec, x_all, y_all, z_all, L, N, N_perp):
        matr = np.zeros((3 * L, 3 * L), dtype = 'complex')
        for i in range(L):
                print('tau3:', i)
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
                                        shift = nx * a_vec + ny * b_vec
                                        x_cur = x + shift[0]
                                        y_cur = y + shift[1]
                                        z_cur = z + shift[2]
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

# def q(struct, params, theta, E0 = 0.03):
#         print("Start...")
#         struct, _ = rotate_structure_optical_axis_to_x(struct)
#         a_vec, b_vec, c_vec = get_lattice_vectors(struct)
#         x_all, y_all, z_all, types_all = replicate_along_z(struct, N)
#         L = len(types_all)
#         ### реплицирование, потом передавать уже нормальные ячейки
#         tau_1 = tau1(a_vec, b_vec, x_all, y_all, z_all, L)
#         tau_2 = tau2(a_vec, b_vec, x_all, y_all, z_all, L)
#         tau_3 = tau3(a_vec, b_vec, x_all, y_all, z_all, L)
#         M_matr = M(params, types_all, L)
#         K_matr = K(params, types_all, L)
#         A = -omega**2 * M_matr + K_matr - tau_1 -1j * omega * tau_2 + omega **2 * tau_3
#         A_inv = np.linalg.inv(A)
#         b = np.zeros(3 * L, dtype = 'complex')
#         theta_rad = theta / 180 * np.pi
#         for i in range(L):
#               z = z_all[i]
#               phase = np.exp(-1j * omega * z / c)
#               b[3 * i] = E0 * np.cos(theta_rad) * phase
#               b[3 * i + 1] = E0 * np.sin(theta_rad) * phase
#               b[3 * i + 2] = 0
#         q = A_inv @ b
#         return q

def q_mod(a_vec, b_vec, x_all, y_all, z_all, types_all, theta = 0.0, E0 = 0.03, N = 100, N_perp = 15, omega = 3e15):
        print("Это q_mod")

        L = len(types_all)
        ### реплицирование, потом передавать уже нормальные ячейки
        
        tau_1 = tau1(a_vec, b_vec, x_all, y_all, z_all, L, N, N_perp)
        tau_2 = tau2(a_vec, b_vec, x_all, y_all, z_all, L, N, N_perp)
        tau_3 = tau3(a_vec, b_vec, x_all, y_all, z_all, L, N, N_perp)
        M_matr = M(params, types_all, L)
        K_matr = K(params, types_all, L)
        A = -omega**2 * M_matr + K_matr - tau_1 -1j * omega * tau_2 + omega **2 * tau_3
        b = np.zeros(3 * L, dtype = 'complex')
        theta_rad = theta / 180 * np.pi
        for i in range(L):
              z = z_all[i]
              phase = np.exp(-1j * omega * z / c)
              b[3 * i] = E0 * np.cos(theta_rad) * phase
              b[3 * i + 1] = E0 * np.sin(theta_rad) * phase
              b[3 * i + 2] = 0
        q = np.linalg.solve(A, b)
        print(f"norm(E) = {np.linalg.norm(b)}")
        print(f"norm(tau_1 q) = {np.linalg.norm(tau_1 @ q)}")
        print(f"max tau_1 q = {np.max(tau_1 @ q)}")
        print(f"norm(omega * tau_2 q) = {omega * np.linalg.norm(tau_2 @ q)}")
        print(f"max omega * tau_2 q = {omega * np.max(tau_2 @ q)}")
        print(f"norm(omega^2 * tau_3 q) = {omega**2 * np.linalg.norm(tau_3 @ q)}")
        print(f"max(omega^2 * tau_3 q) = {omega**2 * np.max(tau_3 @ q)}")
        print(f"norm(K * q) = {np.linalg.norm(K_matr @ q)}")
        print(f"norm(omega^2 * M_matr q) = {omega**2 * np.linalg.norm(M_matr @ q)}")
        return q

def phase_analysis(q_vec, x_all, y_all, z_all, S, idx = 0):
        L = len(q_vec) // 3
        N_cells = L // S
        q_atoms = q_vec.reshape(L, 3)

        q_components = q_atoms.reshape(N_cells, S, 3)
        phases = np.angle(q_components[:,idx,:])

        z_grid = z_all.reshape(N_cells, S)
        x_grid = x_all.reshape(N_cells, S)
        
        coords_z = z_grid[:, idx]
        coords_x = x_grid[:, idx]
        return phases, coords_x, coords_z

def n(q_vec, x_all, y_all, z_all, S, idx = 0):
    L = len(q_vec) // 3
    N_cells = L // S
    q_atoms = q_vec.reshape(L, 3)

    q_components = q_atoms.reshape(N_cells, S, 3)
    n_np = np.angle(q_components[:,idx,:])

    z_grid = z_all.reshape(N_cells, S)
    x_grid = x_all.reshape(N_cells, S)
    
    coords_z = z_grid[:, idx]
    coords_x = x_grid[:, idx]
    

N = 10
N_perp = 10
omega = 9e17
cif_path = '/home/ubun/projects/light-propagation-in-single-crystals/md-simulation/output/unit_cells/MgS.cif'
struct = Structure.from_file(cif_path)
S_elements = len(struct)

struct, _ = rotate_structure_optical_axis_to_x(struct)
a_vec, b_vec, c_vec = get_lattice_vectors(struct)

x_all, y_all, z_all, types_all = replicate_along_z(struct, N)
print(z_all)
print(struct)


t1 = time.time()
q_ans = q_mod(a_vec, b_vec, x_all, y_all, z_all, types_all, N, N_perp, omega)
t2 = time.time()
print(f"Time: {t2 - t1}")


phases, coords_x, coords_z = phase_analysis(q_ans, x_all, y_all, z_all, S_elements)
phases1, coords_x1, coords_z1 = phase_analysis(q_ans, x_all, y_all, z_all, S_elements, idx = 1)
print(phases[:,0])
print(f"q_ans: {q_ans}")
print(f"coords_z: {coords_z}")
N_np = np.arange(N)
ref_np = -omega * coords_z / c
#Посмотреть на реплицированную ячейку


plt.errorbar(coords_z, coords_x, fmt = '.')
plt.errorbar(coords_z1, coords_x1, fmt = '.')
plt.grid()
plt.show()

### Фазы

plt.plot(coords_z, phases[:,0] % np.pi, label = 'x')
plt.plot(coords_z, ref_np % np.pi, label = 'ref')
plt.errorbar(coords_z, phases[:,0] % np.pi, fmt = '.')
# plt.plot(N_np, phases[:,1], label = 'y')
# plt.plot(N_np, phases[:,2], label = 'z')
plt.legend()
plt.grid()
plt.show()


                

                    






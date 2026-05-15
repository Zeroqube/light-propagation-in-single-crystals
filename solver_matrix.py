import numpy as np
from pymatgen.core import Structure

cif_path = '/home/ubun/projects/light-propagation-in-single-crystals/md-simulation/output/unit_cells/CaCO3.cif'
struct = Structure.from_file(cif_path)

params = {
        'Ca' : {'m' : 40, 'k' : 1.0},
        'O' : {'m' : 16, 'k' : 1.0},
}


N_perp = 1 # количество слоёв вверх и столько же вниз

N = 1 # кол-во ячеек вдоль направления распространения

x_0_arr = []
y_0_arr = []
z_0_arr = []
types = []

for site in struct:
        x, y, z = site.coords
        atom_type = site.species_string
        
        x_0_arr.append(x)
        y_0_arr.append(y)
        z_0_arr.append(z)
        types.append(atom_type)

print(types)

print(x_0_arr)



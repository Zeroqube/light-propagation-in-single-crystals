#!/bin/bash
# Запуск LAMMPS для одного материала.
# Использование:
#   sbatch run.sh GaN              — обычный запуск
#   sbatch --array=0-7 run.sh      — массив; материал берётся из MATERIALS[SLURM_ARRAY_TASK_ID]
# Скрипт стартует из md-simulation/, читает inputs/in.<material>, пишет в dumps/ и logs/.

#SBATCH -J LAMMPS
#SBATCH -N 1
#SBATCH -t 800
#SBATCH -p small

#SBATCH -o logs/lammps_%x_%A_%a.out
#SBATCH -e logs/lammps_%x_%A_%a.err

###SBATCH --gres=gpu:1
###SBATCH --reservation=test

export OMP_NUM_THREADS=16
###export OMP_PROC_BIND=spread

# --- Окружение ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lammps-kim

# --- Выбор материала ---
# Список готовых к запуску материалов (status=kim в potentials.csv)
MATERIALS=(GaN SiC Al2O3 SiO2 TiO2 CdS ZnO BN)

if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
    MATERIAL="${MATERIALS[${SLURM_ARRAY_TASK_ID}]}"
elif [ -n "$1" ]; then
    MATERIAL="$1"
else
    echo "Укажи материал: ./run.sh <Material>  или  sbatch --array=0-$((${#MATERIALS[@]}-1)) run.sh"
    exit 1
fi

INPUT="inputs/in.${MATERIAL}"
LOG="logs/log.${MATERIAL}"

if [ ! -f "${INPUT}" ]; then
    echo "Не найден ${INPUT}. Сначала запусти gen_inputs.py."
    exit 1
fi

echo "=== ${MATERIAL} | $(date) ==="
echo "input: ${INPUT}"
echo "log:   ${LOG}"
lmp -in "${INPUT}" -log "${LOG}"
echo "=== ${MATERIAL} done | $(date) ==="

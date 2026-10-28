#!/bin/bash
#SBATCH -A advance
#SBATCH -p cp6
#SBATCH -N 40
#SBATCH -n 2240
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH -J acm3d_sphere_restart_check
#SBATCH -o /fs2/home/mrz/DNDSR-Kirabo/logs/%x-%j.out
#SBATCH -e /fs2/home/mrz/DNDSR-Kirabo/logs/%x-%j.err

# Before submission: mkdir -p /fs2/home/mrz/DNDSR-Kirabo/logs
# Submit using sbatch inside the same cluster environment as the original job.
# This script initializes the restart with ZERO steps and no flow-field output.
set -eo pipefail
trap 'printf "FAILED at line %s, exit=%s\n" "$LINENO" "$?" >&2' ERR

source /fs2/software/modules/bashrc
module purge
module load python/3.13_miniforge3-26.1.1
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /fs2/home/mrz/DNDSR-Kirabo/.conda-py312
module load GCC/12.2.0
module load MPI/mpich/4.0.2-mpi-x-gcc12.2
set -u

: "${SLURM_JOB_ID:?Submit this script with sbatch; do not execute it directly on the login node.}"
PROJECT=/fs2/home/mrz/DNDSR-Kirabo
EXE="$PROJECT/build/app/acm3D.exe"
CONFIG="$PROJECT/cases/acm/3D/acm3D_sphere_Re100_1M_restart10000_add20000.json"
RUN_LOG="$PROJECT/logs/sphere_restart_check-${SLURM_JOB_ID}.log"

test -x "$EXE"
test -r "$CONFIG"
python - "$CONFIG" <<'PYCONFIG'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
restart = config['restartSettings']
assert restart['writerRanks'] == 2240, 'Expected writerRanks=2240'
assert int(os.environ['SLURM_NTASKS']) == 2240, 'Expected 2240 allocated tasks'
assert restart['completedSteps'] == 10000, 'Expected completedSteps=10000'
for label, name in [('mesh', config['meshSettings']['meshFile']), ('restart', restart['flowFile'])]:
    path = Path(name)
    if not path.is_file() or not os.access(path, os.R_OK):
        raise RuntimeError(f'{label} file missing or unreadable: {path}')
    print(f'{label}: {path} ({path.stat().st_size} bytes)', flush=True)
print(f'Configuration: {config_path}', flush=True)
print('Preflight passed. Only restart initialization will run.', flush=True)
PYCONFIG

cd "$PROJECT/build"
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=true
module list 2>&1
printf 'Combined solver log: %s\n' "$RUN_LOG"

# Capture both streams before MPI cancellation messages obscure the first error.
set +e
yhrun -n 2240 -c 1 --ntasks-per-node=56 \
  "$EXE" "$CONFIG" \
  -k /timeMarchSettings/nSteps -v 0 \
  -k /outputSettings/writeInitial -v false \
  -k /outputSettings/interval -v 0 \
  > "$RUN_LOG" 2>&1
run_status=$?
set -e
printf 'Restart check exit code: %s\n' "$run_status"
tail -n 100 "$RUN_LOG"
exit "$run_status"

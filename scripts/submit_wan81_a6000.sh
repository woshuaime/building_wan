#!/bin/bash
#SBATCH --job-name=building_wan_a6000
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=192G
#SBATCH --time=10-00:00:00
#SBATCH --comment=videoedit2025
#SBATCH --output=/mnt/windowsE/chuanjun/building_wan/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/windowsE/chuanjun/building_wan/logs/slurm/%x-%j.err

set -euo pipefail

cd "${BUILDING_WAN_ROOT:-/mnt/windowsE/chuanjun/building_wan}"
bash scripts/train_wan81_a6000.sh

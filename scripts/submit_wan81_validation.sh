#!/bin/bash
#SBATCH --job-name=building_wan_val
#SBATCH --account=u2025171963
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --comment=videoedit2025
#SBATCH --output=/home/share/CHUANJUN/building_wan/logs/slurm/%x-%j.out
#SBATCH --error=/home/share/CHUANJUN/building_wan/logs/slurm/%x-%j.err

set -euo pipefail

cd /home/share/CHUANJUN/building_wan
bash scripts/validate_wan81.sh

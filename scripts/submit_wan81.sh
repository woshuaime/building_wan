#!/bin/bash
#SBATCH --job-name=building_wan
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=192G
#SBATCH --time=10-00:00:00
#SBATCH --comment=videoedit2025
#SBATCH --output=/home/share/CHUANJUN/building_wan/logs/slurm/%x-%j.out
#SBATCH --error=/home/share/CHUANJUN/building_wan/logs/slurm/%x-%j.err

set -euo pipefail

cd /home/share/CHUANJUN/building_wan
bash scripts/train_wan81.sh

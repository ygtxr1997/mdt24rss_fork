#!/bin/bash
#SBATCH -J mdt24rss
#SBATCH -p High
#SBATCH -N 1
#SBATCH --ntasks-per-node=4          # differs from T2V that only 1 task per dist per node!
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=24           # number of cores per tasks
#SBATCH --mem=300000MB               # memory
#SBATCH -t 72:00:00
#SBATCH --comment=MDT24RSS
#SBATCH --output=/public/home/group_xudong/gavinyuan/code/mdt24rss_fork/slurm/logs/%x-%j.out

set -x -e

export PYTHONPATH=${PWD}

# Add task here
srun --jobid $SLURM_JOBID bash -c 'bash slurm/sh_train.sh'

echo The end time is: `date +"%Y-%m-%d %H:%M:%S"`


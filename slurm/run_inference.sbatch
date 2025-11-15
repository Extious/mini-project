#!/bin/bash
#SBATCH --job-name=gpu-infer-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge
module load miniconda cuda/12.1

source activate gpu-infer
cd $SLURM_SUBMIT_DIR

torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=${SLURM_GPUS_PER_NODE} \
  -m src.cli \
  --model gpt2 \
  --dataset synthetic \
  --mode distributed \
  --batch-size 96 \
  --num-samples 4096 \
  --precision fp16 \
  --max-length 512 \
  --profile \
  --profile-dir data/profiler/slurm_ddp \
  --metrics-path data/slurm_distributed.json

#!/bin/bash
#SBATCH --job-name=sst2-peft-lora-tinybert
#SBATCH --output=results/logs/out-%j.txt
#SBATCH --error=results/logs/err-%j.txt
#SBATCH --ntasks=1
#SBATCH --qos=1gpu
#SBATCH --partition=dgx-a100
#SBATCH --gpus=1
#SBATCH --mail-user=ilham.abdillah.alhamdi@gmail.com
#SBATCH --mail-type=ALL

#  Load the key from local environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Variables
IMAGE_NAME=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
PROJECT_ROOT=$(pwd)             

# Create directory for Singularity cache to avoid disk quota issues
export SINGULARITY_TMPDIR=$HOME/temp
mkdir -p $SINGULARITY_TMPDIR
mkdir -p results/logs

# Run the training
# We use 'exec' instead of 'instance start' for batch training scripts
# --nv: Enable GPU
# --bind: Mount your current folder so the code and results are visible
# 3. Execute setup and training in one block
# We use 'bash -c' to run multiple commands inside the container
singularity exec --nv \
    --bind ${PROJECT_ROOT}:/root \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env UV_BREAK_SYSTEM_PACKAGES=1 \
    --pwd /root \
    docker://$IMAGE_NAME \
    bash -c "
        pip install uv && \
        uv pip install --system -r pyproject.toml && \
        uv pip install --system -e . && \
        python -m tinybert_lora.train sst2 \
            args.run_name='dgx-run'
    "
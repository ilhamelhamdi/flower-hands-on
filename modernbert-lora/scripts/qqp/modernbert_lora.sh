#!/bin/bash
#SBATCH --job-name=sst2-modernbert-lora
#SBATCH --output=results/logs/%j/out.txt
#SBATCH --error=results/logs/%j/err.txt
#SBATCH --ntasks=1
#SBATCH --qos=1gpu
#SBATCH --partition=dgx-a100
#SBATCH --gpus=1
#SBATCH --mail-user=ilham.abdillah.alhamdi@gmail.com
#SBATCH --mail-type=ALL

#  Load the key from local environment
if [ -f .env ]; then
    set -a            # Mark all variables for export
    source .env       # Read the file
    set +a            # Turn off auto-export
fi

# Variables
# IMAGE_NAME=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
SIF_PATH=~/singularity_images/pytorch_2.10.0.sif
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
    --pwd /root \
    $SIF_PATH \
    bash -c "
        pip install uv --break-system-packages && \
        uv pip install --system --break-system-packages -r pyproject.toml && \
        uv pip install --system --break-system-packages -e . && \
        python -m modernbert_lora.train qqp \
            args.run_name='dgx-run'
    "
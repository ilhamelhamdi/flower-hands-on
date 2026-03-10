#!/bin/bash
#SBATCH --job-name=sst2-peft-lora-tinybert-docker
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
    set -a            # Mark all variables for export
    source .env       # Read the file
    set +a            # Turn off auto-export
fi

# Variables
IMAGE_NAME=ilhamelhamdi/peft-lora-tinybert:latest
PROJECT_ROOT=$(pwd)

# Create directory for Singularity cache to avoid disk quota issues
export SINGULARITY_TMPDIR=$HOME/temp
mkdir -p $SINGULARITY_TMPDIR
mkdir -p results/logs

# Run the training
# We use 'exec' instead of 'instance start' for batch training scripts
# --nv: Enable GPU
# --bind: Mount your current folder so the code and results are visible
singularity exec --nv \
    --bind ${PROJECT_ROOT}:/root \
    --env WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f2) \
    --pwd /root \
    docker://$IMAGE_NAME \
    python -m tinybert_lora.train sst2 \
    args.run_name="dgx-run" \
    args.num_train_epochs=20
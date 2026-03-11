#!/bin/bash
#SBATCH --job-name=flowertune-modernbert-lora
#SBATCH --output=results/logs/%j/out.txt
#SBATCH --error=results/logs/%j/err.txt
#SBATCH --ntasks=1
#SBATCH --qos=1gpu
#SBATCH --partition=dgx-a100
#SBATCH --gpus=1
#SBATCH --mail-user=ilham.abdillah.alhamdi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00

#  Load the key from local environment
if [ -f .env ]; then
    set -a            # Mark all variables for export
    source .env       # Read the file
    set +a            # Turn off auto-export
fi

# Variables
SIF_PATH=/srv/images/python_3.12.9.sif
PROJECT_ROOT=$(pwd)
INSTANCE_NAME=flowertune_modernbert_trainer      

# Create directory for Singularity cache to avoid disk quota issues
export SINGULARITY_TMPDIR=$HOME/temp
mkdir -p $SINGULARITY_TMPDIR
mkdir -p results/logs

singularity instance start --nv -f \
    --bind ${PROJECT_ROOT}:/root \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --pwd /root \
    $SIF_PATH \
    $INSTANCE_NAME


singularity exec --nv \
    --cwd /root \
    instance://$INSTANCE_NAME \
    bash -c "chmod +x ./setup-env.sh && ./setup-env.sh"

# Keeps the job alive until it reaches the time limit
while true; do sleep 1; done
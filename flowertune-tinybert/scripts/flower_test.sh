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


# Variables
SIF_PATH=/srv/images/python_3.12.9.sif
PROJECT_ROOT=$(pwd)
INSTANCE_NAME=flowertune_modernbert_trainer      

# Create directory for Singularity cache to avoid disk quota issues
export SINGULARITY_TMPDIR=$HOME/temp
mkdir -p $SINGULARITY_TMPDIR
mkdir -p results/logs

singularity instance start --nv \
    --bind ${PROJECT_ROOT}:/root \
    --env-file .env \
    $SIF_PATH \
    $INSTANCE_NAME


singularity exec --nv \
    --cwd /root \
    instance://$INSTANCE_NAME \
    bash -c "
        chmod +x ./setup-env.sh && ./setup-env.sh && \

        flwr config list
    "

# Keeps the job alive until it reaches the time limit
# while true; do sleep 1; done
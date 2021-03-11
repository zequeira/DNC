#!/bin/bash

#SBATCH -D . 								      # Working Directory
#SBATCH -J Noise1 						    # Job Name
#SBATCH --output=./logs_1/%x-%A_%a.log 	    # JobName, JobID, arrayID

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --array=5-13

#SBATCH --time=124:00:00 # expected runtime
#SBATCH --partition=standard

#Job-Status per Mail:
#SBATCH --mail-type=NONE
#SBATCH --mail-user=some.one@tu-berlin.de

source activate $1
srun python -u $2 > ./logs_1/MFCCvsNoise$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID.log $SLURM_ARRAY_TASK_ID

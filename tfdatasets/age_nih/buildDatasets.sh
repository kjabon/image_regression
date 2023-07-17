#!/bin/bash -l

#SBATCH --job-name buildDataset   ## name that will show up in the queue
#SBATCH --cpus-per-task=2  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH -c 2
#SBATCH -o "./slurmLogs/buildDatasets/size/%J.out"
for i in {2..6}
do
   tfds build --config_idx $i

done
wait


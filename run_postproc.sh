#!/bin/bash -l
#PBS -A UMCP0020
#PBS -N postproc
#PBS -o logs/postproc.log
#PBS -e logs/postproc.log
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -l select=1:ncpus=16:mem=80GB:ngpus=0
#PBS -M tchor@umd.edu
#PBS -m abe
#PBS -r n

# Clear the environment from any previously loaded modules
module purge
module load ncarenv/23.10 gcc ncarcompilers netcdf
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

time ~/miniconda3/envs/py310/bin/python h00_run_postproc.py 2>&1 | tee logs/postproc.out

qstat -f $PBS_JOBID >> logs/postproc.out

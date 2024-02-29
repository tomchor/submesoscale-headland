#!/bin/bash -l
#PBS -A UMCP0028
#PBS -N R05F02-f2
#PBS -o logs/NPN-R05F02-f2.log
#PBS -e logs/NPN-R05F02-f2.log
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l gpu_type=a100
#PBS -M tchor@umd.edu
#PBS -m abe

# Clear the environment from any previously loaded modules
module --force purge
module load ncarenv/23.10 gcc ncarcompilers netcdf
module load julia/1.9 cuda
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

export JULIA_DEPOT_PATH="/glade/work/tomasc/.julia"
echo $CUDA_VISIBLE_DEVICES

time julia --project --pkgimages=no regular_headland.jl --simname=NPN-R05F02-f2 2>&1 | tee logs/NPN-R05F02-f2.out
#time julia --check-bounds=no --pkgimages=no --project headland.jl --simname=NPN-R05F02-f2 2>&1 | tee logs/NPN-R05F02-f2.out

qstat -f $PBS_JOBID >> logs/NPN-R05F02-f2.log
qstat -f $PBS_JOBID >> logs/NPN-R05F02-f2.out


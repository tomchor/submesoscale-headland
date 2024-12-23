from os import system

#+++ Define simnames
simnames = [#"NPN-TEST",
            "NPN-R008F008",
            "NPN-R02F008",
            "NPN-R05F008",
            "NPN-R1F008",
            "NPN-R008F02",
            "NPN-R02F02",
            "NPN-R05F02",
            "NPN-R1F02",
            "NPN-R008F05",
            "NPN-R02F05",
            "NPN-R05F05",
            "NPN-R1F05",
            "NPN-R008F1",
            "NPN-R02F1",
            "NPN-R05F1",
            "NPN-R1F1",
            ]

from cycler import cycler
names = cycler(name=simnames)
resolutions = cycler(resolution = ["-f4", "-f2", ""])
resolutions = cycler(resolution = ["-f2", ""])
modifiers = cycler(rotation = ["", "-S", "-AMD"])
modifiers = cycler(rotation = ["", "-S"])
simnames = [ nr["name"] + nr["rotation"] + nr["resolution"] for nr in modifiers * resolutions * names ]
#---

#+++ Options
remove_checkpoints = False
only_one_job = False
dry_run = False
omit_topology = True

topology_string = "NPN-"
verbose = 1
aux_filename = "aux_pbs_twake.sh"
julia_file = "headland.jl"
#---

pbs_script = \
"""#!/bin/bash -l
#PBS -A PROJECTCODE
#PBS -N {simname_fullshort}
#PBS -o logs/{simname_full}.log
#PBS -e logs/{simname_full}.log
#PBS -l walltime=23:59:00
#PBS -q casper
#PBS -l {options_string1}
#PBS -l {options_string2}
#PBS -M email@univ.edu
#PBS -m ae
#PBS -r n

# Clear the environment from any previously loaded modules
module li
module --force purge
module load ncarenv/23.10
module load julia/1.10.2 cuda
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

export JULIA_DEPOT_PATH="/glade/work/tomasc/.julia"
echo $CUDA_VISIBLE_DEVICES

time julia --project --pkgimages=no {julia_file} --simname={simname} 2>&1 | tee logs/{simname_full}.out
#time julia --check-bounds=no --pkgimages=no --project {julia_file} --simname={simname} 2>&1 | tee logs/{simname_full}.out

qstat -f $PBS_JOBID >> logs/{simname_full}.log
qstat -f $PBS_JOBID >> logs/{simname_full}.out
"""

for simname in simnames:

    #+++ Define simulation name
    simname_full = f"{simname}"

    if omit_topology and simname_full.startswith(topology_string):
        simname_fullshort = simname_full.replace(topology_string, "")
    else:
        simname_fullshort = simname_full
    #----

    #++++ Remove previous checkpoints
    if remove_checkpoints:
        cmd0 = f"rm data/chk.{simname_full}*.jld2"
        if verbose>0: print(cmd0)
        system(cmd0)
    #---

    #+++ Fill pbs script and define command
    options1 = dict(select=1, ncpus=1, ngpus=1)
    options2 = dict()

    if ("-f16" in simname) or ("-f8" in simname) or ("-f4" in simname):
        options2 = options2 | dict(gpu_type = "v100")
        cmd1 = f"qsub {aux_filename}"
    elif ("-f2" in simname):
        options2 = options2 | dict(gpu_type = "a100")
        cmd1 = f"qsub {aux_filename}"
    else:
        options1 = options1 | dict(cpu_type = "milan")
        options2 = options2 | dict(gpu_type = "a100")

        if only_one_job:
            cmd1 = f"qsub {aux_filename}"
        else:
            if "F008" in simname:
                cmd1 = f"JID1=`qsub {aux_filename}`; JID2=`qsub -W depend=afterok:$JID1 {aux_filename}`; JID3=`qsub -W depend=afterok:$JID2 {aux_filename}`; qrls $JID1"
            else:
                cmd1 = f"JID1=`qsub {aux_filename}`; JID2=`qsub -W depend=afterok:$JID1 {aux_filename}`; qrls $JID1"

    options_string1 = ":".join([ f"{key}={val}" for key, val in options1.items() ])
    options_string2 = ":".join([ f"{key}={val}" for key, val in options2.items() ])

    pbs_script_filled = pbs_script.format(simname_fullshort=simname_fullshort, simname=simname,
                                          simname_full=simname_full, julia_file=julia_file,
                                          options_string1=options_string1, options_string2=options_string2)
    if verbose>1: print(pbs_script_filled)
    if verbose>0: print(cmd1)
    #---

    #+++ Run command
    if not dry_run:
        with open(aux_filename, "w") as f:
            f.write(pbs_script_filled)
        system(cmd1)
    #---

    print()

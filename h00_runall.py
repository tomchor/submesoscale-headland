import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import subprocess
import numpy as np
import xarray as xr
import pynanigans as pn
from aux00_utils import open_simulation, condense, check_simulation_completion
from aux01_physfuncs import get_topography_masks
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar

print("Starting h00 script")

#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames_base = [#"NPN-TEST",
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
modifiers = ["-f4", "-S-f4", "-f2", "-S-f2"]
modifiers = ["-f4", "-f2", ""]
#---


for modifier in modifiers:
    simnames = [ simname_base + modifier for simname_base in simnames_base ]
    check_simulation_completion(simnames, slice_name="ttt", path=path)

for modifier in modifiers:
    print("\nStarting h01 and h02 post-processing of results using modifier", modifier)
    simnames = [ simname_base + modifier for simname_base in simnames_base ]
    exec(open("h01_energy_transfer.py").read())
    exec(open("h02_bulkstats.py").read())

print("\nStarting hvid00")

#+++ Options for hvid00
parallel = True
animate = True
test = False
time_avg = False
summarize = False
zoom = False
plotting_time = 23
figdir = "figures_check"

slice_names = ["xyi",]

varnames = ["PV_norm", "Ro", "εₖ"]
varnames = ["PV_norm",]
contour_variable_name = None #"water_mask_buffered"
#---

aux_modifiers = [ modifier.replace("-", "", 1) if modifier != "" else "f1" for modifier in modifiers ]
arglist = ["--parallel" if parallel else "",
           "--animate" if animate else "",
           "--test" if test else "",
           "--time_avg" if time_avg else "",
           "--summarize" if summarize else "",
           "--zoom" if zoom else "",
           "--plotting_time", str(plotting_time),
           "--figdir", figdir,
           "--slice_names", *slice_names,
           "--varnames", *varnames,
           "--modifiers", *aux_modifiers,
           "--contour_variable_name" if contour_variable_name is not None else "",
           ]
arglist = [ item for item in arglist if item != "" ]
print("Using arguments", arglist)
subprocess.run(["/glade/u/home/tomasc/miniconda3/envs/py310/bin/python", "hvid00_facetgrid.py", *arglist])

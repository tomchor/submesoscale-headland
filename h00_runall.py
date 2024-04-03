import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
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
modifiers = ["-f2", "-S-f2"]
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

for modifier in modifiers:
    print("\nStarting h03 post-processing of results using modifier", modifier)
    simnames = [ simname_base + modifier for simname_base in simnames_base ]
    exec(open("h03_collect_snapshots.py").read())

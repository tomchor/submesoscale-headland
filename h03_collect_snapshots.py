import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
import pynanigans as pn
from aux00_utils import collect_datasets, open_simulation, adjust_times
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar

print("Starting snapshot-collecting script")

#+++ Options
slice_names = ["xyi", "xiz", "iyz", "tafields"]
#---

#+++ Define collect_bulkstats()
def collect_bulkstats(simnames_filtered)
    dslist = []
    for sim_number, simname in enumerate(simnames_filtered):
        fname = f"{bulkname}_{simname}.nc"
        print(f"\nOpening {fname}")
        ds = xr.open_dataset(f"data_post/{fname}", chunks=dict(time="auto", L="auto"))

        ds["simulation"] = simname
        ds["sim_number"] = sim_number
        ds = ds.expand_dims(("Ro_h", "Fr_h")).assign_coords(Ro_h=[np.round(ds.Ro_h, decimals=4)],
                                                            Fr_h=[np.round(ds.Fr_h, decimals=4)])
        dslist.append(ds.reset_coords())
    dsout = xr.combine_by_coords(dslist, combine_attrs="drop_conflicts")

    #+++ save to disk
    outname = f"data_post/{bulkname}_snaps{modifier}.nc"
    with ProgressBar():
        print(f"Saving results to {outname}")
        dsout.to_netcdf(outname)
    #---
    return
#---

#+++ Define directory and simulation name
if basename(__file__) != "h00_runall.py":
    path = f"./headland_simulations/data/"
    simnames = ["NPN-R008F008",
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

    modifiers = ["-f4", "-S-f4", "-f2", "-S-f2", "", "-S"]
    modifiers = ["-f4", "-f2", ""]

    for modifier in modifiers:
        simnames_filtered = [ f"{simname}{modifier}" for simname in simnames ]
        collect_bulkstats(simnames_filtered)

        for slice_name in slice_names:
            snaps = collect_datasets(simnames_filtered, slice_name=slice_name)

            #+++ Save snapshots to disk
            outname = f"data_post/{slice_name}_snaps{modifier}.nc"
            with ProgressBar():
                print(f"Saving results to {outname}")
                snaps.to_netcdf(outname)
            #---
else:
    simnames_filtered = simnames
    collect_bulkstats(simnames_filtered)

    for slice_name in slice_names:
        snaps = collect_datasets(simnames_filtered, slice_name=slice_name)

        #+++ Save snapshots to disk
        outname = f"data_post/{slice_name}_snaps{modifier}.nc"
        with ProgressBar():
            print(f"Saving results to {outname}")
            snaps.to_netcdf(outname)
        #---
#---

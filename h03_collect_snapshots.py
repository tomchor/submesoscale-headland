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

#+++ Define collect_and_save_datasets() function
def collect_and_save_datasets():

    #+++ Collect 2D slices
    for slice_name in slice_names:
        print(f"\nOpening slice {slice_name}")

        dslist = []
        for sim_number, simname in enumerate(simnames_filtered):
            #+++ Open datasets
            #+++ Deal with time-averaged output
            if slice_name == "tafields":
                fname = f"tafields_{simname}.nc"
                print(f"\nOpening {fname}")
                ds = xr.open_dataset(f"data_post/{fname}", chunks=dict(time="auto", L="auto"))
            #---

            #+++ Deal with snapshots
            else:
                fname = f"{slice_name}.{simname}.nc"
                print(f"\nOpening {fname}")
                ds = open_simulation(path + fname,
                                     use_advective_periods=True,
                                     topology=simname[:3],
                                     squeeze=True,
                                     load=False,
                                     get_grid = False,
                                     open_dataset_kwargs=dict(chunks=dict(time=1)),
                                     )

                if slice_name == "xyi":
                    ds = ds.drop_vars(["zC", "zF"])
                    variables = ["u", "v", "w", "Ro", "PV", "εₖ", "εₚ", "Δxᶜᶜᶜ", "Δyᶜᶜᶜ", "Δzᶜᶜᶜ",
                                 "∂u∂x", "∂v∂x", "∂w∂x",
                                 "∂u∂y", "∂v∂y", "∂w∂y",
                                 "∂u∂z", "∂v∂z", "∂w∂z",
                                 "dbdx", "dbdy", "dbdz",
                                 "∂Uᵍ∂z", "∂Vᵍ∂z", "PV_z", "Re_b"]
                elif slice_name == "xiz":
                    ds = ds.drop_vars(["yC", "yF"])
                    variables = ["u", "v", "w", "dbdz", "Ro", "PV", "εₖ", "εₚ", "Δxᶜᶜᶜ", "Δyᶜᶜᶜ", "Δzᶜᶜᶜ", "b",]
                elif slice_name == "iyz":
                    ds = ds.drop_vars(["xC", "xF"])
                    variables = ["u", "v", "w", "dbdz", "Ro", "PV", "εₖ", "εₚ", "Δxᶜᶜᶜ", "Δyᶜᶜᶜ", "Δzᶜᶜᶜ", "b",]
                ds = ds[variables]

                #+++ Get rid of slight misalignment in times
                ds = adjust_times(ds, round_times=True)
                #---

                #+++ Get specific times and create new variables
                t_slice = slice(ds.T_advective_spinup+10, np.inf, 1)
                ds = ds.sel(time=t_slice)
                ds = ds.assign_coords(time=ds.time-ds.time[0])

                #+++ Get a unique list of time (make it as long as the longest ds
                if not dslist:
                    time_values = np.array(ds.time)
                else:
                    if len(np.array(ds.time)) > len(time_values):
                           time_values = np.array(ds.time)
                #---
                #---
            #---
            #---

            #+++ Calculate resolutions before they get thrown out
            ds["Δx_min"] = ds["Δxᶜᶜᶜ"].where(ds["Δxᶜᶜᶜ"] > 0).min().values
            ds["Δy_min"] = ds["Δyᶜᶜᶜ"].where(ds["Δyᶜᶜᶜ"] > 0).min().values
            ds["Δz_min"] = ds["Δzᶜᶜᶜ"].where(ds["Δzᶜᶜᶜ"] > 0).min().values
            #---

            #+++ Create auxiliary variables and organize them into a Dataset
            if "PV" in ds.variables.keys():
                ds["PV_norm"] = ds.PV / (ds.N2_inf * ds.f_0)
            ds["simulation"] = simname
            ds["sim_number"] = sim_number
            ds["f₀"] = ds.f_0
            ds["N²∞"] = ds.N2_inf
            ds = ds.expand_dims(("Ro_h", "Fr_h")).assign_coords(Ro_h=[np.round(ds.Ro_h, decimals=4)],
                                                                Fr_h=[np.round(ds.Fr_h, decimals=4)])
            dslist.append(ds)
            #---

        #+++ Create snapshots dataset
        for i, ds in enumerate(dslist[1:]):
            try:
                if slice_name == "xyi":
                    assert np.allclose(dslist[0].yC.values, ds.yC.values), "y coordinates don't match in all datasets"
                elif slice_name == "xiz":
                    assert np.allclose(dslist[0].zC.values, ds.zC.values), "z coordinates don't match in all datasets"
            except AttributeError:
                if slice_name == "xyi":
                    assert np.allclose(dslist[0].y.values, ds.y.values), "y coordinates don't match in all datasets"
                elif slice_name == "xiz":
                    assert np.allclose(dslist[0].z.values, ds.z.values), "z coordinates don't match in all datasets"
            if "time" in ds.coords.keys():
                assert np.allclose(dslist[0].time.values, ds.time.values), "Time coordinates don't match in all datasets"

        print("Starting to concatenate everything into one dataset")
        if slice_name != "tafields":
            for i in range(len(dslist)):
                dslist[i]["time"] = time_values # Prevent double time, e.g. [0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.8] etc. (not sure why this is needed)
        dsout = xr.combine_by_coords(dslist, combine_attrs="drop_conflicts")

        dsout["Δxᶜᶜᶜ"] = dsout["Δxᶜᶜᶜ"].isel(Ro_h=0, Fr_h=0)
        dsout["Δyᶜᶜᶜ"] = dsout["Δyᶜᶜᶜ"].isel(Ro_h=0, Fr_h=0)
        dsout["Δzᶜᶜᶜ"] = dsout["Δzᶜᶜᶜ"].isel(Ro_h=0, Fr_h=0)
        dsout["land_mask"]  = (dsout["Δxᶜᶜᶜ"] == 0)
        dsout["water_mask"] = np.logical_not(dsout.land_mask)
        #---

        #+++ Save to disk
        outname = f"data_post/{slice_name}_snaps{modifier}.nc"

        with ProgressBar():
            print(f"Saving results to {outname}")
            dsout.to_netcdf(outname)
        #---
    #---

    #+++ Now collect bulkstats
    #for bulkname in ["bulkstats", "bulkstats_xyz"]:
    for bulkname in ["bulkstats",]:
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
    simnames = [#"NPN-TEST",
                "NPN-R008F008",
                "NPN-R008F02",
                "NPN-R008F05",
                "NPN-R008F1",
                "NPN-R02F008",
                "NPN-R02F02",
                "NPN-R02F05",
                "NPN-R02F1",
                "NPN-R05F008",
                "NPN-R05F02",
                "NPN-R05F05",
                "NPN-R05F1",
                "NPN-R1F008",
                "NPN-R1F02",
                "NPN-R1F05",
                "NPN-R1F1",
                ]

    modifiers = ["-f4", "-S-f4", "-f2", "-S-f2", "", "-S"]
    modifiers = ["-f4", "-f2",]
    modifiers = ["-f4",]

    for modifier in modifiers:
        simnames_filtered = [ f"{simname}{modifier}" for simname in simnames ]
        collect_and_save_datasets()
else:
    simnames_filtered = simnames
    collect_and_save_datasets()
#---

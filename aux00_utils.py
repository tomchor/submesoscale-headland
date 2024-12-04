import xarray as xr
import pynanigans as pn
import numpy as np

#+++ All simulation names
simnames = ["NPN-R008F008",
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

threed_sims = ["R1F1",
               "R05F05",
               ]

bathfo_sims = ["R008F1",
               "R008F05",
               "R02F1",
               ]

vertco_sims = ["R008F008",
               "R02F02",
               ]

vertsh_sims = ["R05F008",
               "R1F008",
               "R1F02",
               ]
#---

#+++ Open simulation following the standard way
def open_simulation(fname, 
                    use_inertial_periods=False,
                    use_cycle_periods=False,
                    use_advective_periods=False,
                    use_strouhal_periods=False,
                    open_dataset_kwargs=dict(),
                    regularize_ds_kwargs=dict(),
                    load=False,
                    squeeze=True,
                    unique=True,
                    verbose=False,
                    get_grid = True,
                    topology="PPN", **kwargs):
    if verbose: print(sname, "\n")
    
    #+++ Open dataset and create grid before squeezing
    if load:
        ds = xr.load_dataset(fname, decode_times=False, **open_dataset_kwargs)
    else:
        ds = xr.open_dataset(fname, decode_times=False, **open_dataset_kwargs)
    #---

    #+++ Get grid
    if get_grid: grid_ds = pn.get_grid(ds, topology=topology, **kwargs)
    #---

    #+++ Squeeze?
    if squeeze: ds = ds.squeeze()
    #---

    #+++ Normalize units and regularize
    if use_inertial_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_inertial, new_units="Inertial period")
    elif use_advective_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_advective, new_units="Cycle period")
    elif use_cycle_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_cycle, new_units="Cycle period")
    elif use_strouhal_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_strouhal, new_units="Strouhal period")
    #---

    #+++ Returning only unique times:
    if unique:
        import numpy as np
        _, index = np.unique(ds['time'], return_index=True)
        if verbose and (len(index)!=len(ds.time)): print("Cleaning non-unique indices")
        ds = ds.isel(time=index)
    #---

    #+++ Return
    if get_grid:
        return grid_ds, ds
    else:
        return ds
    #---
#---

#+++ Condense variables into one (in datasets)
def condense(ds, vlist, varname, dimname="α", indices=None):
    """
    Condense variables in `vlist` into one variable named `varname`.
    In the process, individual variables in `vlist` are removed from `ds`.
    """
    if indices is None:
        indices = range(1, len(vlist)+1)

    ds[varname] = ds[vlist].to_array(dim=dimname).assign_coords({dimname : list(indices)})
    ds = ds.drop(vlist)
    return ds
#---

#+++ Time adjustment
def adjust_times(ds, round_times=True, decimals=4):
    import numpy as np
    from statistics import mode
    Δt = np.round(mode(ds.time.diff("time").values), decimals=5)
    ds = ds.sel(time=np.arange(ds.time[0], ds.time[-1]+Δt/2, Δt), method="nearest")

    if round_times:
        ds = ds.assign_coords(time = list( map(lambda x: np.round(x, decimals=decimals), ds.time.values) ))
    return ds
#---

#+++ Check if all simulations are complete
def check_simulation_completion(simnames, slice_name="ttt", path="./headland_simulations/data/"):
    from colorama import Fore, Back, Style
    times = []
    for simname in simnames:
        with open_simulation(path+f"{slice_name}.{simname}.nc", use_advective_periods = True, get_grid = False) as ds:
            ds = adjust_times(ds, round_times=True)
            times.append(ds.time.values)
            print(simname, ds.time.values)
    message = Fore.GREEN + "All times equal" + Style.RESET_ALL
    for time in times[1:]:
        if (len(time)!=len(times[0])) or  (time != times[0]).any():
            message = Fore.RED + "Not all times are equal!" + Style.RESET_ALL
    print(message)
    return
#---

#+++ Define collect_datasets() function
def collect_datasets(simnames_filtered, slice_name="xyi", path="./headland_simulations/data/", verbose=False):
    dslist = []
    for sim_number, simname in enumerate(simnames_filtered):
        #+++ Open datasets
        #+++ Deal with time-averaged output
        if slice_name == "tafields":
            fname = f"tafields_{simname}.nc"
            print(f"\nOpening {fname}")
            ds = xr.open_dataset(f"data_post/{fname}", chunks=dict(time="auto", L="auto"))
        #---

        #+++ Deal with volume-integrated output
        elif slice_name == "bulkstats":
            fname = f"bulkstats_{simname}.nc"
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
            elif slice_name == "xiz":
                ds = ds.drop_vars(["yC", "yF"])
            elif slice_name == "iyz":
                ds = ds.drop_vars(["xC", "xF"])

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
        if "Δx_min" not in ds.keys(): ds["Δx_min"] = ds["Δxᶜᶜᶜ"].where(ds["Δxᶜᶜᶜ"] > 0).min().values
        if "Δy_min" not in ds.keys(): ds["Δy_min"] = ds["Δyᶜᶜᶜ"].where(ds["Δyᶜᶜᶜ"] > 0).min().values
        if "Δz_min" not in ds.keys(): ds["Δz_min"] = ds["Δzᶜᶜᶜ"].where(ds["Δzᶜᶜᶜ"] > 0).min().values
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
            if verbose:
                for ds in dslist:
                    print(ds.time)
            assert np.allclose(dslist[0].time.values, ds.time.values), "Time coordinates don't match in all datasets"

    print("Starting to concatenate everything into one dataset")
    if slice_name != "tafields" and slice_name != "bulkstats":
        for i in range(len(dslist)):
            dslist[i]["time"] = time_values # Prevent double time, e.g. [0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.8] etc. (not sure why this is needed)
    dsout = xr.combine_by_coords(dslist, combine_attrs="drop_conflicts")

    if "Δxᶜᶜᶜ" in ds.keys():
        dsout["Δxᶜᶜᶜ"] = dsout["Δxᶜᶜᶜ"].isel(Ro_h=0, Fr_h=0)
        dsout["land_mask"]  = (dsout["Δxᶜᶜᶜ"] == 0)
        dsout["water_mask"] = np.logical_not(dsout.land_mask)
    if "Δyᶜᶜᶜ" in ds.keys(): dsout["Δyᶜᶜᶜ"] = dsout["Δyᶜᶜᶜ"].isel(Ro_h=0, Fr_h=0)
    if "Δzᶜᶜᶜ" in ds.keys(): dsout["Δzᶜᶜᶜ"] = dsout["Δzᶜᶜᶜ"].isel(Ro_h=0, Fr_h=0)
    #---

    return dsout
#---

#+++ Downsample / chunk
def down_chunk(ds, max_time=np.inf, **kwargs):
    ds = ds.sel(time=slice(0, max_time))
    ds = pn.downsample(ds, **kwargs)
    ds = ds.pnchunk(maxsize_4d=1000**2, round_func=np.ceil)
    return ds
#---

#+++ Simulation names and filters
wake_sims = [#"NPN-TEST-f8",
             "R008F008",
             #"R008F02",
             #"R008F05",
             #"R008F1",
             "R02F008",
             "R02F02",
             #"R02F05",
             #"R02F1",
             "R05F008",
             "R05F02",
             "R05F05",
             #"R05F1",
             "R1F008",
             "R1F02",
             "R1F05",
             "R1F1",
             ]
#---

#+++ Make names match the paper
def prettify_names(ds):
    return ds.assign_coords(simulation=[ pnames[sim] for sim in ds.simulation.values ])

def simplify_names(ds):
    return ds.assign_coords(simulation=[ sim.replace("PPN-", "").replace("PNN-", "") for sim in ds.simulation.values ])
#---

#+++ Simulation filters by name
def filter_by_resolution(simname, resolution):
    import re
    pattern = r"-f\d+"
    match = re.search(pattern, simname)
    if match:
        if (resolution != "") and (resolution in simname):
            return True
    else:
        if (resolution == ""):
            return True
    return False

def is_wake_sim(simname):
    """ Returns True if simulation is expected to have detached wakes"""
    return any([ wake_sim in simname for wake_sim in wake_sims ])

def all_wake_sims(simnames):
    """ Returns all simulations that areexpected to have detached wakes"""
    return [ simname for simname in simnames if is_wake_sim(simname) ]

def filter_sims(ds, prettify=True, only_main=False):
    if only_main:
        simulations = list(filter(lambda x: x in main_sims, ds.simulation.values))
        ds = ds.reindex(simulation=simulations)

    if prettify:
        ds = prettify_names(ds)

    return ds
#---

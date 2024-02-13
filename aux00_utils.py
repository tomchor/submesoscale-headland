import xarray as xr
import pynanigans as pn
import numpy as np

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
                    topology="PPN", **kwargs):
    if verbose: print(sname, "\n")
    
    #++++ Open dataset and create grid before squeezing
    if load:
        ds = xr.load_dataset(fname, decode_times=False, **open_dataset_kwargs)
    else:
        ds = xr.open_dataset(fname, decode_times=False, **open_dataset_kwargs)
    grid_ds = pn.get_grid(ds, topology=topology, **kwargs)
    #----

    #++++ Squeeze?
    if squeeze: ds = ds.squeeze()
    #----

    #++++ Normalize units and regularize
    if use_inertial_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_inertial, new_units="Inertial period")
    elif use_advective_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_advective, new_units="Cycle period")
    elif use_cycle_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_cycle, new_units="Cycle period")
    elif use_strouhal_periods:
        ds = pn.normalize_time_by(ds, seconds=ds.T_strouhal, new_units="Strouhal period")
    #----

    #++++ Returning only unique times:
    if unique:
        import numpy as np
        _, index = np.unique(ds['time'], return_index=True)
        if verbose and (len(index)!=len(ds.time)): print("Cleaning non-unique indices")
        ds = ds.isel(time=index)
    #----

    return grid_ds, ds
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

#+++ Downsample / chunk
def down_chunk(ds, max_time=np.inf, **kwargs):
    ds = ds.sel(time=slice(0, max_time))
    ds = pn.downsample(ds, **kwargs)
    ds = ds.pnchunk(maxsize_4d=1000**2, round_func=np.ceil)
    return ds
#---

#+++ Simulation names and filters
pnames = {"PPN-R1F002A15" : "R1F002A15",
          "PPN-R01F02A10" : "R01F02A10",
          "PPN-R02F02A10" : "R02F02A10",
          "PPN-R02F002A10" : "R02F002A10",
          }

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

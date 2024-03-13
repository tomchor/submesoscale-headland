import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
import pynanigans as pn
from aux00_utils import open_simulation, condense
from aux01_physfuncs import get_topography_masks
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
π = np.pi

print("Starting bulk statistics script")

#+++ Define directory and simulation name
if basename(__file__) != "h00_runall.py":
    path = f"./headland_simulations/data/"
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
    modifiers = cycler(modifier = ["-f4", "-S-f4", "-f2", "-S-f2", "", "-S"])
    modifiers = cycler(modifier = ["-f4",])
    simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

outnames = []
for simname in simnames:
    #+++ Open datasets
    print(f"\nOpening {simname}")
    grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time="auto")),
                                    )
    grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time="auto")),
                                    )
    tafields = xr.open_dataset(f"data_post/tafields_{simname}.nc", decode_times=False, chunks="auto")
    #---

    #+++ Preliminary definitions and checks
    if simname.startswith("NPN"):
        pass
    else:
        raise NameError

    Δt = xyi.time.diff("time").median()
    Δt_tol = Δt/100
    if np.all(xyi.time.diff("time") > Δt_tol):
        print(Fore.GREEN + f"Δt is consistent for {simname}", Style.RESET_ALL)
    else:
        print(f"Δt is inconsistent for {simname}")
        print(np.count_nonzero(xyi.time.diff("time") < Δt_tol), "extra time steps")

        tslice1 = slice(0.0, None, None)
        xyi = xyi.sel(time=tslice1)

        xyz = xyz.reindex(dict(time=np.arange(0, xyi.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
        xyi = xyi.reindex(dict(time=np.arange(0, xyi.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
    #---

    #+++ Pre-processing before averaging
    t_slice = slice(tafields.T_advective_spinup+0.01, np.inf)

    xyi = xyi.reindex_like(tafields).sel(time=t_slice).drop_dims(("xF", "yF"))
    xyz = xyz.reindex_like(tafields).sel(time=t_slice).drop_dims(("xF", "yF", "zF"))
    #---

    #+++ Get means from tafields integrals
    bulk = xr.Dataset()
    bulk["∫∫∫ᵇ1dxdydz"] = tafields["∫∫∫ᵇ1dxdydz"]
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "w̄b̄", "⟨w′b′⟩ₜ", "⟨wb⟩ₜ", "⟨Ek′⟩ₜ"]:
        int_var = f"∫∫∫ᵇ{var}dxdydz"
        bulk[int_var] = tafields[int_var]
        bulk[f"⟨{var}⟩ᵇ"] = bulk[int_var] / bulk["∫∫∫ᵇ1dxdydz"]

    bulk["∫∫ᵇ1dxdz"] = tafields["∫∫ᵇ1dxdz"]
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "⟨Ek′⟩ₜ"]:
        int_var = f"∫∫ᵇ{var}dxdz"
        bulk[int_var] = tafields[int_var]
        bulk[f"⟨{var}⟩ˣᶻ"] = bulk[int_var] / bulk["∫∫ᵇ1dxdz"]

    bulk["∫∫∫ᵋ1dxdydz"] = tafields["∫∫∫ᵋ1dxdydz"]
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "⟨wb⟩ₜ"]:
        int_var = f"∫∫∫ᵋ{var}dxdydz"
        bulk[int_var] = tafields[int_var]
        bulk[f"⟨{var}⟩ᵋ"] = bulk[int_var] / bulk["∫∫∫ᵋ1dxdydz"]

    bulk["∫∫ᶜˢⁱ1dxdy"] = tafields["∫∫ᶜˢⁱ1dxdy"]
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR"]:
        int_var = f"∫∫ᶜˢⁱ{var}dxdy"
        bulk[int_var] = tafields[int_var]
        bulk[f"⟨{var}⟩ᶜˢⁱ"] = bulk[int_var] / bulk["∫∫ᶜˢⁱ1dxdy"]

    bulk["⟨Π⟩ᵇ"] = bulk["⟨SPR⟩ᵇ"].sum("j")
    bulk["⟨Π⟩ˣᶻ"] = bulk["⟨SPR⟩ˣᶻ"].sum("j")
    bulk["⟨Π⟩ᵋ"] = bulk["⟨SPR⟩ᵋ"].sum("j")
    #---

    #+++ Create auxiliaty variables and organize them into a Dataset
    bulk.attrs = tafields.attrs

    bulk["Slope_Bu"] = bulk.Slope_Bu
    bulk["Ro_h"] = bulk.Ro_h
    bulk["Fr_h"] = bulk.Fr_h
    bulk["α"] = bulk.α

    bulk["Bu_h"] = bulk.Bu_h
    bulk["Γ"] = bulk.Γ
    bulk["c_dz"] = bulk.c_dz

    bulk["f₀"] = bulk.f_0
    bulk["N²∞"] = bulk.N2_inf
    bulk["V∞"] = bulk.V_inf
    bulk["L"] = bulk.L
    bulk["headland_intrusion_size"] = bulk.headland_intrusion_size_max
    bulk["Δx_min"] = tafields["Δxᶜᶜᶜ"].where(tafields["Δxᶜᶜᶜ"] > 0).min().values
    bulk["Δy_min"] = tafields["Δyᶜᶜᶜ"].where(tafields["Δyᶜᶜᶜ"] > 0).min().values
    bulk["Δz_min"] = tafields["Δzᶜᶜᶜ"].where(tafields["Δzᶜᶜᶜ"] > 0).min().values

    bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h
    bulk["V∞³÷L"] = bulk.V_inf**3 / bulk.L
    bulk["V∞²N∞"] = bulk.V_inf**2 * np.sqrt(bulk.N2_inf)
    bulk["N∞³L²"] = np.sqrt(bulk.N2_inf)**3 * bulk.L**2

    bulk["Kb"]  = -bulk["⟨⟨wb⟩ₜ⟩ᵇ"] / bulk["N²∞"]
    bulk["Kb′"] = -bulk["⟨⟨w′b′⟩ₜ⟩ᵇ"] / bulk["N²∞"]
    bulk["Kb̄"]  = -bulk["⟨w̄b̄⟩ᵇ"] / bulk["N²∞"]
    bulk["Kbᵋ"] = -bulk["⟨⟨wb⟩ₜ⟩ᵋ"] / bulk["N²∞"]

    bulk["Loᵋ"] = 2*π * np.sqrt(bulk["⟨ε̄ₖ⟩ᵋ"] / bulk.N2_inf**(3/2))
    #---

    #+++ Final touches and save
    bulk = bulk.reset_coords()

    outname = f"data_post/bulkstats_{simname}.nc"
    outnames.append(outname)

    with ProgressBar(minimum=2, dt=5):
        print(f"Saving results to {outname}...")
        bulk.to_netcdf(outname)
    #---

#+++ Collect everything
if basename(__file__) == "h00_runall.py":
    dslist = []
    print()
    for sim_number, outname in enumerate(outnames):
        print(f"Opening {outname}")
        ds = xr.open_dataset(outname, chunks=dict(time="auto", L="auto"))

        ds["simulation"] = simname
        ds["sim_number"] = sim_number
        ds = ds.expand_dims(("Ro_h", "Fr_h")).assign_coords(Ro_h=[np.round(ds.Ro_h, decimals=4)],
                                                            Fr_h=[np.round(ds.Fr_h, decimals=4)])
        dslist.append(ds.reset_coords())
        ds.close()

    print("Combining outputs...", end="")
    dsout = xr.combine_by_coords(dslist, combine_attrs="drop_conflicts")
    print("done")

    outname_snaps = f"data_post/bulkstats_snaps{modifier}.nc"
    with ProgressBar():
        print(f"Saving results to {outname_snaps}")
        dsout.to_netcdf(outname_snaps)
#---

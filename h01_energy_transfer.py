import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
import pynanigans as pn
from aux00_utils import open_simulation, condense, adjust_times
from colorama import Fore, Back, Style
from dask.diagnostics import ProgressBar
xr.set_options(display_width=140, display_max_rows=30)

print("Starting energy transfer script")

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

    from cycler import cycler
    names = cycler(name=simnames)
    modifiers = cycler(modifier = ["-f4", "-S-f4", "-f2", "-S-f2", "", "-S"])
    modifiers = cycler(modifier = ["-f4",])
    simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

#+++ Options
indices = [1, 2, 3]
#---

for simname in simnames:
    #+++ Open datasets xyz and xyi
    print(f"\nOpening {simname} xyz")
    grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    print(f"Opening {simname} xyi")
    grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    print(f"Opening {simname} ttt")
    grid_ttt, ttt = open_simulation(path+f"ttt.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    print(f"Opening {simname} tti")
    grid_tti, tti = open_simulation(path+f"tti.{simname}.nc",
                                    use_advective_periods = True,
                                    topology = simname[:3],
                                    squeeze = True,
                                    load = False,
                                    open_dataset_kwargs = dict(chunks="auto"),
                                    )
    #---

    #+++ Get rid of slight misalignment in times
    xyz = adjust_times(xyz, round_times=True)
    xyi = adjust_times(xyi, round_times=True)
    ttt = adjust_times(ttt, round_times=True)
    tti = adjust_times(tti, round_times=True)
    #---

    #+++ Preliminary definitions and checks
    print("Doing prelim checks")
    if simname.startswith("NPN"):
        pass
    else:
        raise NameError

    Δt = xyz.time.diff("time").median()
    Δt_tol = Δt/100
    if np.all(xyi.time.diff("time") > Δt_tol):
        print(Fore.GREEN + f"Δt is consistent for {simname}", Style.RESET_ALL)
    else:
        print(f"Δt is inconsistent for {simname}")
        print(np.count_nonzero(xyz.time.diff("time") < Δt_tol), "extra time steps")

        tslice1 = slice(0.0, None, None)
        xyz = xyz.sel(time=tslice1)

        xyi = xyi.reindex(dict(time=np.arange(0, xyz.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
        xyz = xyz.reindex(dict(time=np.arange(0, xyz.time[-1]+1e-5, Δt)), method="nearest", tolerance=Δt/Δt_tol)
    #---

    #+++ Trimming domain
    t_slice = slice(ttt.T_advective_spinup+0.01, np.inf)
    x_slice = slice(xyz.xF[0], np.inf)
    y_slice = slice(xyz.yF[0] + xyz.sponge_length_y, np.inf)
    z_slice = slice(ttt.zF[0], ttt.zC[-1])

    xyz = xyz.sel(time=t_slice, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice, zC=z_slice, zF=z_slice)
    xyi = xyi.sel(time=t_slice, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    ttt = ttt.sel(time=t_slice, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    tti = tti.sel(time=t_slice, xC=x_slice, xF=x_slice, yC=y_slice, yF=y_slice)
    #---

    #+++ Condense tensors
    def condense_velocities(ds):
        return condense(ds, ["u", "v", "w"], "uᵢ", dimname="i", indices=indices)

    def condense_velocity_gradient_tensor(ds):
        ds = condense(ds, ["∂u∂x", "∂v∂x", "∂w∂x"], "∂₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂y", "∂v∂y", "∂w∂y"], "∂₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂u∂z", "∂v∂z", "∂w∂z"], "∂₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["∂₁uᵢ", "∂₂uᵢ", "∂₃uᵢ"], "∂ⱼuᵢ", dimname="j", indices=indices)
        return ds

    def condense_geostrophic_velocity_gradient_tensor(ds):
        ds = condense(ds, ["∂Uᵍ∂x", "∂Vᵍ∂x",], "∂₁Uᵍᵢ", dimname="i", indices=[1, 2])
        ds = condense(ds, ["∂Uᵍ∂y", "∂Vᵍ∂y",], "∂₂Uᵍᵢ", dimname="i", indices=[1, 2])
        ds = condense(ds, ["∂Uᵍ∂z", "∂Vᵍ∂z",], "∂₃Uᵍᵢ", dimname="i", indices=[1, 2])
        ds = condense(ds, ["∂₁Uᵍᵢ", "∂₂Uᵍᵢ", "∂₃Uᵍᵢ"], "∂ⱼUᵍᵢ", dimname="j", indices=indices)
        ds["∂ⱼUᵍᵢ"].loc[dict(i=3)] = 0 # Geostrophic w velocities are zero!
        return ds

    def condense_reynolds_stress_tensor(ds):
        ds["vu"] = ds.uv
        ds["wv"] = ds.vw
        ds["wu"] = ds.uw
        ds = condense(ds, ["uu",   "uv",   "uw"],   "u₁uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["vu",   "vv",   "vw"],   "u₂uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["wu",   "wv",   "ww"],   "u₃uᵢ", dimname="i", indices=indices)
        ds = condense(ds, ["u₁uᵢ", "u₂uᵢ", "u₃uᵢ"], "uⱼuᵢ", dimname="j", indices=indices)
        return ds

    ttt = condense_velocities(ttt)
    ttt = condense_velocity_gradient_tensor(ttt)
    ttt = condense_reynolds_stress_tensor(ttt)
    tti = condense_velocities(tti)
    tti = condense_velocity_gradient_tensor(tti)
    tti = condense_geostrophic_velocity_gradient_tensor(tti)
    tti = condense(tti, ["dbdx", "dbdy", "dbdz"], "∂ⱼb", dimname="j", indices=indices)
    #---

    #+++ Time average
    # Here ū and ⟨u⟩ₜ are interchangeable
    tafields = ttt.mean("time")
    tafields = tafields.rename({"uᵢ"   : "ūᵢ",
                                "∂ⱼuᵢ" : "∂ⱼūᵢ",
                                "uⱼuᵢ" : "⟨uⱼuᵢ⟩ₜ",
                                "b"    : "b̄",
                                "uᵢGᵢ" : "⟨wb⟩ₜ",
                                "εₖ"   : "ε̄ₖ",
                                "εₚ"   : "ε̄ₚ",
                                })
    tafields.attrs = ttt.attrs
    #---

    #+++ Get turbulent Reynolds stress tensor
    tafields["ūⱼūᵢ"]     = tafields["ūᵢ"] * tafields["ūᵢ"].rename(i="j")
    tafields["⟨u′ⱼu′ᵢ⟩ₜ"] = tafields["⟨uⱼuᵢ⟩ₜ"] - tafields["ūⱼūᵢ"]
    #---

    #+++ Get shear production rates
    tafields["SPR"]  = - (tafields["⟨u′ⱼu′ᵢ⟩ₜ"] * tafields["∂ⱼūᵢ"]).sum("i")
    #---

    #+++ Get buoyancy production rates
    tafields["w̄b̄"]     = tafields["ūᵢ"].sel(i=3) * tafields["b̄"]
    tafields["⟨w′b′⟩ₜ"] = tafields["⟨wb⟩ₜ"] - tafields["w̄b̄"]
    #---

    #+++ Get TKE
    tafields["⟨Ek′⟩ₜ"] = (tafields["ūⱼūᵢ"].sel(i=1, j=1) + tafields["ūⱼūᵢ"].sel(i=2, j=2) + tafields["ūⱼūᵢ"].sel(i=3, j=3)) / 2
    #---

    #+++ Volume-average/integrate results so far
    tafields["ΔxΔyΔz"] = tafields["Δxᶜᶜᶜ"] * tafields["Δyᶜᶜᶜ"] * tafields["Δzᶜᶜᶜ"]
    def integrate(da, dV=tafields["ΔxΔyΔz"], dims=("x", "y", "z")):
        return (da*dV).pnsum(dims)

    tafields["1"] = xr.ones_like(tafields["Δxᶜᶜᶜ"])
    buffer = 5 # meters
    distance_mask = tafields.altitude > buffer
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "w̄b̄", "⟨w′b′⟩ₜ", "⟨wb⟩ₜ", "⟨Ek′⟩ₜ", "1"]:
        int_all = f"∫∫∫⁰{var}dxdydz"
        int_buf = f"∫∫∫⁵{var}dxdydz"
        tafields[int_all] = integrate(tafields[var])
        tafields[int_buf] = integrate(tafields[var], dV=tafields.ΔxΔyΔz.where(distance_mask))
        tafields = condense(tafields, [int_all, int_buf], f"∫∫∫ᵇ{var}dxdydz", dimname="buffer", indices=[0, buffer])

    tafields["average_turbulence_mask"] = tafields["ε̄ₖ"] > 1e-11
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "⟨wb⟩ₜ", "1"]:
        int_turb = f"∫∫∫ᵋ{var}dxdydz"
        tafields[int_turb] = integrate(tafields[var], dV=tafields.ΔxΔyΔz.where(tafields.average_turbulence_mask))
    #---

    #+++ Get time-avg results at half-depth
    tafields = tafields.sel(zC=tti.zC.item(), method="nearest")
    tafields["q̄z"] = tti.PV_z.mean("time")
    tafields["q̄"] = tti.PV.mean("time")

    tafields["∂ⱼūᵢ"]  = tti["∂ⱼuᵢ"].mean("time")
    tafields["∂ⱼŪᵍᵢ"] = tti["∂ⱼUᵍᵢ"].mean("time")
    tafields["∂ⱼb̄"]   = tti["∂ⱼb"].mean("time")
    tafields["R̄e_b"]  = tti["Re_b"].mean("time")

    tafields["⟨uᵢ∂ⱼuⱼuᵢ⟩ₜ"] = tti["uᵢ∂ⱼuⱼuᵢ"].mean("time")
    tafields["⟨uᵢ∂ᵢp⟩ₜ"]    = tti["uᵢ∂ᵢp"].mean("time")
    tafields["⟨uᵢ∂ⱼτᵢⱼ⟩ₜ"]  = tti["uᵢ∂ⱼτᵢⱼ"].mean("time")
    tafields["⟨uᵢ∂ⱼτᵇᵢⱼ⟩ₜ"] = tti["uᵢ∂ⱼτᵇᵢⱼ"].mean("time")
    #---

    #+++ Get CSI mask and CSI-integral
    tafields["average_CSI_mask"] = ((tafields.q̄ * tafields.f_0) < 0) & (tafields["∂ⱼb̄"].sel(j=3) > 0)
    ΔxΔy = tafields["Δxᶜᶜᶜ"] * tafields["Δyᶜᶜᶜ"]
    for var in ["ε̄ₖ", "ε̄ₚ", "SPR", "1"]:
        int_csi = f"∫∫ᶜˢⁱ{var}dxdy"
        tafields[int_csi] = integrate(tafields[var], dV=ΔxΔy.where(tafields.average_CSI_mask), dims=("x", "y"))
    #---

    #+++ Drop unnecessary vars
    tafields = tafields.drop_vars(["ūⱼūᵢ",
                                   "⟨uⱼuᵢ⟩ₜ", "⟨u′ⱼu′ᵢ⟩ₜ",
                                   "ΔxΔyΔz",
                                   "ΔxC", "ΔyC", "ΔzC"])
    tafields = tafields.drop_dims(("xF", "yF", "zF"))
    #---

    #+++ Save
    outname = f"data_post/tafields_{simname}.nc"
    with ProgressBar(minimum=5, dt=5):
        print(f"Saving results to {outname}...")
        tafields.to_netcdf(outname)
        print("Done!\n")
    xyi.close(); xyz.close(); ttt.close(); tti.close()
    #---

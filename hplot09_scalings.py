import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from scipy.optimize import curve_fit

modifier = ""

tafields = xr.open_dataset(f"data_post/tafields_snaps{modifier}.nc", decode_times=False)

bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h))).mean("yC")

#+++ Define new variables
bulk["γᵇ"] = bulk["⟨ε̄ₚ⟩ᵇ"] / (bulk["⟨ε̄ₚ⟩ᵇ"] + bulk["⟨ε̄ₖ⟩ᵇ"])
bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h
bulk["RoRi"] = bulk.Ro_h / bulk.Fr_h**2

bulk["∫∫∫ᵇΠdxdydz"] = bulk["⟨Π⟩ᵇ"] * bulk["∫∫∫ᵇ1dxdydz"]

bulk["⟨ε̄ₖ⟩ᴮᴸ"] = bulk["⟨ε̄ₖ⟩ᵇ"].sel(buffer=0) - bulk["⟨ε̄ₖ⟩ᵇ"]
bulk["εₖ_ratio_bl_to_rest"] = bulk["⟨ε̄ₖ⟩ᴮᴸ"] / bulk["⟨ε̄ₖ⟩ᵇ"]

bulk["⟨Πᶻ⟩"] = bulk["⟨SPR⟩ᵇ"].sel(j=3)
bulk["SP_ratio1"] = bulk["⟨SPR⟩ᵇ"].sel(j=[1,2]).sum("j") / bulk["⟨SPR⟩ᵇ"].sel(j=3)
bulk["SP_ratio3"] = bulk["∫∫ᶜˢⁱSPRdxdy"].sel(j=[1,2]).sum("j") / bulk["∫∫ᶜˢⁱSPRdxdy"].sel(j=3)
#---

#+++ Choose buffers and set some attributes
bulk.RoFr.attrs = dict(long_name="$Ro_h Fr_h$")
bulk.RoRi.attrs = dict(long_name="$Ro_h / Fr_h^2$")
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
#---

for buffer in bulk.buffer.values:
    print(f"Plotting with buffer = {buffer} m")
    bulk_buff = bulk.sel(buffer=buffer)

    #+++ Create figure
    ncols = 3
    nrows = 3
    size = 3.5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize = (1.4*ncols*size, nrows*size),
                             sharex=False, sharey=False,
                             constrained_layout=True)
    axesf = axes.flatten()
    #---

    #+++ Marker details
    marker_large_Bu = "^"
    marker_unity_Bu = "s"
    marker_small_Bu = "o"
    markers = [marker_large_Bu, marker_unity_Bu, marker_small_Bu]

    color_large_Bu = "blue"
    color_unity_Bu = "orange"
    color_small_Bu = "green"
    colors = [color_large_Bu, color_unity_Bu, color_small_Bu]
    
    conditions = [bulk_buff.Bu_h>1, bulk_buff.Bu_h==1, bulk_buff.Bu_h<1]
    labels = ["Bu>1", "Bu=1", "Bu<1"]
    #---

    #+++ Auxiliary continuous variables
    RoFr = np.logspace(np.log10(bulk_buff.RoFr.min())+1/2, np.log10(bulk_buff.RoFr.max())-1/2)
    S_Bu = np.logspace(np.log10(bulk_buff["Slope_Bu"].min())+1/3, np.log10(bulk_buff["Slope_Bu"].max())-1/3)
    #---

    #+++ Plot stuff
    print("Plotting axes 0")
    ax = axesf[0]
    xvarname = "RoFr"
    yvarname = "Kb"
    bulk_buff["Kb"].attrs = dict(long_name=r"$K_b = \langle wb \rangle / N^2_\infty$")
    ax.set_title(bulk_buff[yvarname].attrs["long_name"])
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(RoFr, 1e-2*RoFr, ls="--", label=r"$Ro_h Fr_h$")
    ax.plot(RoFr, 2e-2*RoFr**2.0, ls="--", label=r"$(Ro_h Fr_h)^2$")

    print("Plotting axes 1")
    ax = axesf[1]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇε̄ₖdxdydz"
    ax.scatter(x=bulk_buff[xvarname], y=bulk_buff[yvarname], label="", color="k")
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, 7e-4*S_Bu, ls="--", label=r"Slope_Bu")
    ax.plot(S_Bu, 7e-4*S_Bu**(1/2), ls="--", label=r"Slope_Bu$^{1/2}$")

    print("Plotting axes 2")
    ax = axesf[2]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇε̄ₚdxdydz"
    ax.scatter(x=bulk_buff[xvarname], y=bulk_buff[yvarname], label="", color="k")
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, 2e-4*S_Bu, ls="--", label=r"Slope_Bu")
    ax.plot(S_Bu, 2e-4*S_Bu**(1/2), ls="--", label=r"Slope_Bu$^{1/2}$")

    print("Plotting axes 3")
    ax = axesf[3]
    xvarname = "RoFr"
    yvarname = "γᵇ"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0, .5)

    print("Plotting axes 4")
    ax = axesf[4]
    bulk_buff.SP_ratio3.attrs = dict(long_name=r"$\langle \Pi^h \rangle^c / \langle \Pi^v \rangle^c$")
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇΠdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_xlabel(xvarname); ax.set_ylabel(yvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, 7e-4*S_Bu, ls="--", label=r"Slope_Bu")
    ax.plot(S_Bu, 7e-4*S_Bu**(1/2), ls="--", label=r"Slope_Bu$^{1/2}$")



    print("Plotting axes 5")
    ax = axesf[5]
    bulk_buff["Fr"] = bulk_buff.Fr_h + 0*bulk_buff.Ro_h
    xvarname = "Fr"
    yvarname = "SP_ratio1"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_xlabel(xvarname); ax.set_ylabel(yvarname)
    ax.set_xscale("log"); ax.set_yscale("log")


    print("Plotting axes 6")
    ax = axesf[6]
    xvarname = "RoFr"
    yvarname = "Kb̄"
    bulk_buff["Kb̄"].attrs = dict(long_name=r"$K_b = \overline{w}\overline{b} / N^2_\infty$")
    ax.set_title(bulk_buff[yvarname].attrs["long_name"])
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(RoFr, 1e-2*RoFr, ls="--", label=r"$Ro_h Fr_h$")
    ax.plot(RoFr, 2e-2*RoFr**2.0, ls="--", label=r"$(Ro_h Fr_h)^2$")
 

    print("Plotting axes 7")
    ax = axesf[7]
    xvarname = "RoFr"
    yvarname = "Kb′"
    bulk_buff["Kb′"].attrs = dict(long_name=r"$K_b = \langle w`b` \rangle / N^2_\infty$")
    ax.set_title(bulk_buff[yvarname].attrs["long_name"])
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(RoFr, 1e-2*RoFr, ls="--", label=r"$Ro_h Fr_h$")
    ax.plot(RoFr, 2e-2*RoFr**2.0, ls="--", label=r"$(Ro_h Fr_h)^2$")



    print("Plotting axes 8")
    ax = axesf[8]
    xvarname = "RoFr"
    yvarname = "Kbᵋ"
    bulk_buff["Kbᵋ"].attrs = dict(long_name=r"$K_b = \langle w`b` \rangle^\epsilon / N^2_\infty$")
    ax.set_title(bulk_buff[yvarname].attrs["long_name"])
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(yvarname); ax.set_xlabel(xvarname)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(RoFr, 1e-2*RoFr, ls="--", label=r"$Ro_h Fr_h$")
    ax.plot(RoFr, 2e-2*RoFr**2.0, ls="--", label=r"$(Ro_h Fr_h)^2$")


 
    #print("Plotting axes 5")
    #ax = axesf[5]
    #xvarname = "Slope_Bu"
    #yvarname = "∫∫ʷuᵢ∂ⱼτᵇᵢⱼdxdz"
    #for cond, label, color, marker in zip(conditions, labels, colors, markers):
    #    ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    #ax.set_xlabel(xvarname); ax.set_ylabel(yvarname)
    #ax.set_xscale("log"); ax.set_yscale("log")
    #ax.plot(S_Bu, 2e-4*S_Bu, ls="--", label=r"Slope_Bu")
    #ax.plot(S_Bu, 2e-4*S_Bu**(1/2), ls="--", label=r"Slope_Bu$^{1/2}$")
    #
    #
    #print("Plotting axes 6")
    #ax = axesf[6]
    #xvarname = "Slope_Bu"
    #yvarname = "⟨u∇τᵇ⟩"
    #for cond, label, color, marker in zip(conditions, labels, colors, markers):
    #    ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    #ax.set_xlabel(xvarname); ax.set_ylabel(yvarname)
    #ax.set_xscale("log"); ax.set_yscale("log")
    
    #print("Plotting axes 7")
    #ax = axesf[7]
    #bulk_buff["SP_ratio3"] = bulk_buff["⟨SPR⟩ᶜ"].sel(j=[1,2]).sum("j") / bulk_buff["⟨SPR⟩"].sel(j=3) #bulk_buff["⟨Π⟩ᶜ"]
    #bulk_buff.SP_ratio3.attrs = dict(long_name=r"$\langle \Pi^h \rangle^c / \langle \Pi^v \rangle^c$")
    #xvarname = "SP_ratio3"
    #yvarname = "γᶜ"
    #for cond, label, color, marker in zip(conditions, labels, colors, markers):
    #    ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    #ax.set_xlabel(xvarname); ax.set_ylabel(yvarname)
    #ax.set_ylabel(bulk_buff.SP_ratio3.attrs["long_name"])
    #ax.set_xscale("log"); ax.set_yscale("log")
    #---
    
    #+++ Prettify and save
    for ax in axesf:
        ax.legend()
        ax.grid(True)
        ax.set_title("")
    
    fig.savefig(f"figures_check/scalings_buffer={buffer}m.pdf")
    #---

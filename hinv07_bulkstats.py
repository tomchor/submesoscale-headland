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
bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc")

bulk["γ"]  = bulk["⟨εₚ⟩"]  / (bulk["⟨εₚ⟩"]  + bulk["⟨εₖ⟩"])
bulk["γᵗ"] = bulk["⟨εₚ⟩ᵗ"] / (bulk["⟨εₚ⟩ᵗ"] + bulk["⟨εₖ⟩ᵗ"])
bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h
bulk["RoRi"] = bulk.Ro_h / bulk.Fr_h**2

bulk.RoFr.attrs = dict(long_name="$Ro_h Fr_h$")
bulk.RoRi.attrs = dict(long_name="$Ro_h / Fr_h^2$")
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")


if 1:
    ncols = 2
    nrows = 2
    size = 3.5
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize = (1.4*ncols*size, nrows*size),
                             sharex=False, sharey=False,
                             constrained_layout=True)
    axesf = axes.flatten()

    ax = axesf[0]
    bulk.plot.scatter(ax=ax, x="RoFr", y="Kb", xscale="log", yscale="log", label="", color="k")
    RoFr = np.logspace(np.log10(bulk.RoFr.min())+1/2, np.log10(bulk.RoFr.max())-1/2)
    ax.plot(RoFr, 1e-2*RoFr, ls="--", label=r"$Ro_h Fr_h$")
    ax.plot(RoFr, 1e-2*RoFr**1.5, ls="--", label=r"(Ro_h Fr_h)$^{1/2}$")
    ax.plot(RoFr, 2e-2*RoFr**2.0, ls="--", label=r"(Ro_h Fr_h)$^2$")

    ax = axesf[1]
    bulk.plot.scatter(ax=ax, x="Slope_Bu", y="⟨εₖ⟩", xscale="log", yscale="log", label="", color="k")
    S_Bu = np.logspace(np.log10(bulk["Slope_Bu"].min())+1/3, np.log10(bulk["Slope_Bu"].max())-1/3)
    ax.plot(S_Bu, 2e-11*S_Bu, ls="--", label=r"Slope_Bu")
    ax.plot(S_Bu, 2e-11*S_Bu**(1/2), ls="--", label=r"Slope_Bu$^{1/2}$")
    ax.plot(S_Bu, 2e-11*S_Bu**(2/3), ls="--", label="Slope_Bu$^{2/3}$")

    ax = axesf[2]
    bulk.plot.scatter(ax=ax, x="Slope_Bu", y="⟨εₚ⟩", xscale="log", yscale="log", label="", color="k")
    ax.plot(S_Bu, 2e-11*S_Bu, ls="--", label=r"Slope_Bu")
    ax.plot(S_Bu, 2e-11*S_Bu**(1/2), ls="--", label=r"Slope_Bu$^{1/2}$")
    ax.plot(S_Bu, 2e-11*S_Bu**(2/3), ls="--", label="Slope_Bu$^{2/3}$")

    ax = axesf[3]
    bulk.plot.scatter(ax=ax, x="RoRi", y="γ", xscale="log", label="", color="k")
    ax.set_ylim(0, .5)


    for ax in axesf:
        ax.legend()
        ax.grid(True)
    pause


if False:
    bulk["Kb"].plot(vmin=-1e-3, vmax=1e-3, cmap=cm.balance)
    plt.figure()
    bulk["⟨wb⟩"].plot(vmin=-1e-10, vmax=1e-10, cmap="bwr")
    #plt.figure()
    #bulk["⟨dbdz⟩"].plot(norm=LogNorm())
    #plt.figure()
    #(bulk["⟨dbdz⟩"] / bulk["N²∞"]).plot()
    plt.figure()
    bulk["γ"].plot(vmin=0, vmax=0.8)
    pause



def power_relation(Ro_Fr, a_Ro, b_Fr, β, offset):
    Ro, Fr = Ro_Fr
    return (Ro**a_Ro * Fr**b_Fr) * β + offset

def bilinear_relation(Ro_Fr, a_Ro, b_Fr, lnβ):
    Ro, Fr = Ro_Fr
    return lnβ + a_Ro * Ro + b_Fr * Fr

#variables = ["Kb", "Kbᵗ", "⟨εₖ⟩", "⟨εₖ⟩ᵗ", "⟨εₚ⟩", "⟨εₚ⟩ᵗ", "γ", "γᵗ"]
variables = ["Kb", "⟨εₖ⟩", "⟨εₚ⟩", "γ",]

ncols = min(len(variables), 2)
fig, axes = plt.subplots(ncols=ncols, nrows=int(np.ceil(len(variables) / ncols)), 
                         figsize = (6,6),
                         constrained_layout = True)
axesf = axes.flatten()

for i, variable in enumerate(variables):
    print(variable)
    da = bulk[variable]
    fit_result = np.log(da.where(da>0)).curvefit(("Ro_h", "Fr_h"), bilinear_relation,
                                                 p0 = dict(a_Ro=1, b_Fr=1, lnβ=0),
                                                 kwargs=dict(nan_policy="omit"))
    print(fit_result.curvefit_coefficients)

    a_Ro   = fit_result.curvefit_coefficients[0].values
    b_Fr   = fit_result.curvefit_coefficients[1].values
    β      = np.exp(fit_result.curvefit_coefficients[2]).values
    #offset = fit_result.curvefit_coefficients[3].values

    indep_variable = f"βRoᵃFrᵇ_{variable}"
    bulk[indep_variable] = power_relation((da.Ro_h, da.Fr_h), a_Ro, b_Fr, β, 0)

    ax = axesf[i]
    bulk.plot.scatter(ax=ax, x=indep_variable, y=variable, xscale="log", yscale="log")
    ax.set_title("")
    ax.set_ylabel(variable)
    ax.set_xlabel(f"$\left({β:0.4g}\\right) Ro^{{{a_Ro:0.3f}}} Fr^{{{b_Fr:0.3f}}}$")
    ax.grid(True)
    print()


pause
bulk.plot.scatter(y="Kb",  x="RoFr", label="Kb")
bulk.plot.scatter(y="Kbᵗ", x="RoFr", label="Kbᵗ")

ax = plt.gca()
x_RoFr = [1e-4, 2]
ax.plot(x_RoFr, x_RoFr)

plt.legend()


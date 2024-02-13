import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import open_simulation
from dask.diagnostics import ProgressBar
π = np.pi

#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames = [#"NPN-TEST-f8",
            #"NPN-PropA-f8",
            #"NPN-PropA-f4",
            #"NPN-PropA-f2",
            #"NPN-PropA",
            #"NPN-PropB-f8",
            #"NPN-PropB-f4",
            #"NPN-PropB-f2",
            #"NPN-PropB",
            #"NPN-PropD-f8",
            #"NPN-PropD-f4",
            #"NPN-PropD-f2",
            #"NPN-PropD",
            #"NPN-R02F008-f8",
            #"NPN-R02F008-f4",
            #"NPN-R02F008-f2",
            #"NPN-R02F008",
            #"NPN-R02F02-f8",
            "NPN-R02F02-f4",
            "NPN-R02F02-f2",
            "NPN-R02F02",
            ##"NPN-R02F05-f8",
            #"NPN-R02F05-f4",
            #"NPN-R02F05-f2",
            #"NPN-R02F05",
            ##"NPN-R02F1-f8",
            #"NPN-R02F1-f4",
            #"NPN-R02F1-f2",
            #"NPN-R02F1",
            ##"NPN-R008F02-f8",
            #"NPN-R008F02-f4",
            #"NPN-R008F02-f2",
            #"NPN-R008F02",
            ##"NPN-R05F02-f8",
            #"NPN-R05F02-f4",
            #"NPN-R05F02-f2",
            #"NPN-R05F02",
            ##"NPN-R1F02-f8",
            #"NPN-R1F02-f4",
            #"NPN-R1F02-f2",
            #"NPN-R1F02",
            ##"NPN-R008F008-f8",
            #"NPN-R008F008-f4",
            #"NPN-R008F008-f2",
            #"NPN-R008F008",
            ##"NPN-R05F05-f8",
            #"NPN-R05F05-f4",
            #"NPN-R05F05-f2",
            #"NPN-R05F05",
            ##"NPN-R1F1-f8",
            #"NPN-R1F1-f4",
            #"NPN-R1F1-f2",
            #"NPN-R1F1",
            ]
#---

size = 3.5
n_cycles = 10
dslist = []
for simname in simnames:
    #++++ Open datasets xyz and xyi
    print(f"\nOpening {simname} xyz")
    grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time=1, zC=1)),
                                    )
    print(f"opening {simname} xyi")
    grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=False,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time=1)),
                                    )
    print(f"opening {simname} xiz")
    grid_xiz, xiz = open_simulation(path+f"xiz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=False,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time=1)),
                                    )
    #---

    for ds in [xyi,]:

        #+++ Trimming domain
        t_slice = slice(20, np.inf)
    
        ds = ds.sel(time=t_slice)
        grid_ds = pn.get_grid(ds, topology="NPN")
        #---

        #+++ Interpolations
        ds = ds.chunk(time="auto", xC=-1, yC=-1, zC=-1)
        if len(ds.zC)==1:
            yslice = slice(ds.yF[0] + ds.sponge_length_y, None, 1)
            ds = ds.sel(yC=yslice, yF=yslice)

        elif len(ds.yC)==1:
            ds["dbdz"] = grid_ds.interp(ds.dbdz, "x")
            ds["εₖ"] = grid_ds.interp(grid_ds.interp(ds["εₖ"], "x"), "z")
            ds["Ri"] = grid_ds.interp(ds["Ri"], "x")

        else: # Probabl xyz
            ds["dbdz"] = grid_ds.interp(grid_ds.interp(ds.dbdz, "x"), "y")
            ds["εₖ"] = grid_ds.interp(grid_ds.interp(grid_ds.interp(ds["εₖ"], "x"), "y"), "z")
            ds["Ri"] = grid_ds.interp(grid_ds.interp(ds["Ri"], "x"), "y")

            yslice = slice(0, None, 10)
            xslice = slice(-500, None, 10)
            zslice = slice(None, None, 1)
            ds = ds.sel(xC=xslice, xF=xslice, yC=yslice, yF=yslice, zC=zslice, zF=zslice)

        ds = ds.squeeze()
        #---

        #+++ Create relevant variables
        ds["dbdz_norm"] = ds.dbdz / ds.N2_inf
        ds.dbdz_norm.attrs = dict(long_name = "$\partial b/\partial z$ (normalized)")

        ds["PV_norm"] = ds.PV / (ds.N2_inf * ds.f_0)
        ds.PV_norm.attrs = dict(long_name = "Ertel Potential Vorticity (normalized)")

        ds["PVz_norm"] = ds.PV_z / (ds.N2_inf * ds.f_0)
        ds["PVh_norm"] = (ds.PV_x + ds.PV_y) / (ds.N2_inf * ds.f_0)
        ds["1/Ri"] = 1/ds.Ri
        ds["Ro_tot"] = ds.Ro + 1
        ds["qz_norm"] = ds.PV_z / ds.PV
        ds["q̂z_norm"] = ds.Ro_tot / (ds.Ro_tot - 1/ds.Ri)
        #---

        #+++ Actually plot things
        from matplotlib import pyplot as plt
        plt.rcParams['figure.constrained_layout.use'] = True

        size = 4
        ncols = 2
        nrows = 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(1.2*size*ncols, size*nrows), sharex=False, sharey=False,
                                 squeeze=False)
        axesf = axes.flatten()

        scatter_options = dict(s=4, c=np.log10(ds["εₖ"]), vmin=-10, vmax=-7, cmap="copper_r", rasterized=True)
        heatmap_options = dict(vmin=-1.5, vmax=+1.5, cmap="RdBu_r", rasterized=True)
        contour_options = dict(alpha=0.6, add_colorbar=False, levels=[0], linewidthts=0.5)
        cbar_kwargs = dict(orientation="horizontal", location="top", shrink=0.7, extend="both")

        #+++ Plot phase diagrams
        ax = axes[0,0]
        sc = ax.scatter(x=ds.PV_norm, y=ds.dbdz_norm, **scatter_options)
        plt.colorbar(sc, label="log₁₀(εₖ)", **cbar_kwargs)
        ax.set_ylabel(ds.dbdz_norm.attrs["long_name"])
        ax.set_xlabel(ds.PV_norm.attrs["long_name"])

        ax = axes[1,0]
        ax.scatter(x=ds.PV_norm, y=ds.Ro_tot, **scatter_options)
        ax.set_ylabel("1 + Ro")
        ax.set_xlabel(ds.PV_norm.attrs["long_name"])

        ax = axes[2,0]
        Ro_tot = np.arange(-100, 100)
        Ri_inv = np.arange(-100, 100)
        ax.plot(Ri_inv, Ro_tot, ls="dashed", color="blue", label="PV=0")
        ax.plot(1-Ri_inv, Ro_tot, ls="dotted", color="blue", label="Ro = -Ri⁻¹")
        ax.scatter(x=ds["1/Ri"], y=ds.Ro_tot, **scatter_options)
        ax.set_xlabel("1/Ri")
        ax.set_ylabel("1 + Ro")
        ax.set_ylim(-40, +40)

        ax = axes[3,0]
        PVz_norm = np.arange(-100, 100)
        Ri_inv = np.arange(-100, 100)
        ax.plot(Ri_inv, PVz_norm, ls="dashed", color="blue", label="PV=0")
        ax.plot(1-Ri_inv, PVz_norm, ls="dotted", color="blue", label="PVᶻ = PVʰᵒʳ")
        ax.scatter(x=ds.PVh_norm, y=ds.PVz_norm, **scatter_options)
        ax.set_xlabel("PV (hor.)")
        ax.set_ylabel("PV (vert.)")
        ax.set_ylim(-40, +40)
        #---
        #---

        #+++ Plot snapshots as background
        time = np.inf
        def plot_background(ax, time=np.inf, with_colorbar=True, heatmap_options=heatmap_options):
            if with_colorbar:
                heatmap_options = heatmap_options | dict(add_colorbar=True, cbar_kwargs=cbar_kwargs)
            else:
                heatmap_options = heatmap_options | dict(add_colorbar=False)
            ds.PV_norm.sel(time=time, method="nearest").pnplot(ax=ax, **heatmap_options)
        #---

        #+++ Plot PV-N² phase
        PV_norm_max = 0
        N2_norm_min = 0
        ax = axes[0,1]
        plot_background(ax, time=time, with_colorbar=True)

        mask = np.logical_and(ds.PV_norm < PV_norm_max, ds.dbdz_norm > N2_norm_min)
        contour = mask.sel(time=time, method="nearest").pncontour(ax=ax, colors="red", **contour_options)
        artists, labels = contour.legend_elements()
        ax.legend([artists[0],], ["CSI",], loc="lower left")
        #---

        #+++ Plot PV-Roₜₒₜ phase
        Ro_norm_lim = 0
        ax = axes[1,1]
        plot_background(ax, time=time, with_colorbar=False)

        mask = np.logical_and(ds.PV_norm < PV_norm_max, ds.Ro_tot > Ro_norm_lim)
        contour = mask.sel(time=time, method="nearest").plot.contour(ax=ax, colors="red", alpha=0.6, add_colorbar=False)
        artists_CI, labels = contour.legend_elements()

        mask = np.logical_and(ds.PV_norm < PV_norm_max, ds.Ro_tot < Ro_norm_lim)
        contour = mask.sel(time=time, method="nearest").plot.contour(ax=ax, colors="orange", alpha=0.6, add_colorbar=False)
        artists_SI, labels = contour.legend_elements()
        ax.legend([artists_CI[0], artists_SI[1]], ["CI, SI (sec. Dewar)", "SI only (sec. Dewar)",], loc="lower left")
        #---

        #+++ Plot Ri⁻¹-Roₜₒₜ phase
        ax = axes[2,1]
        plot_background(ax, time=time, with_colorbar=False)

        mask = np.logical_and(ds.PV_norm < 0, ds.Ro < -ds["1/Ri"])
        contour = mask.sel(time=time, method="nearest").plot.contour(ax=ax, colors="red", **contour_options)
        artists_CI, labels = contour.legend_elements()

        mask = np.logical_and(ds.PV_norm < 0, ds.Ro >= -ds["1/Ri"])
        contour = mask.sel(time=time, method="nearest").plot.contour(ax=ax, colors="orange", **contour_options)
        artists_SI, labels = contour.legend_elements()
        ax.legend([artists_CI[0], artists_SI[0]], ["Cen dom", "Sym dom",], loc="lower left")
        #---

        #+++ Plot PVʰᵒʳ-PVᶻ phase
        ax = axes[3,1]
        plot_background(ax, time=time, with_colorbar=False)

        mask = np.logical_and(ds.PV_norm < 0, ds.PVz_norm < ds.PVh_norm)
        contour = mask.sel(time=time, method="nearest").plot.contour(ax=ax, colors="red", **contour_options)
        artists_CI, labels = contour.legend_elements()

        mask = np.logical_and(ds.PV_norm < 0, ds.PVz_norm >= ds.PVh_norm)
        contour = mask.sel(time=time, method="nearest").plot.contour(ax=ax, colors="orange", **contour_options)
        artists_SI, labels = contour.legend_elements()
        ax.legend([artists_CI[0], artists_SI[0]], ["Cen dom", "Sym dom",], loc="lower left")
        #---

        #+++ Prettify axes
        axes[0,0].axhline(y=N2_norm_min, ls="dotted", color="red", alpha=0.6, label="min stratification", zorder=10)
        axes[1,0].axhline(y=Ro_norm_lim, ls="dotted", color="red", alpha=0.6, label="max (1+Ro)", zorder=10)
        for ax in axes[:,0]:
            ax.set_xlim(-40, 40)
            ax.axhline(y=0, ls="--", color="k")
            ax.axvline(x=0, ls="--", color="k")

            ax.axvline(x=PV_norm_max, ls="dotted", color="red", alpha=0.6, label="max PV")
            ax.legend(loc="lower right")

        for ax in axesf:
            ax.grid(True)
            ax.set_title("")
        #---

        #+++ Save figure
        with ProgressBar():
            print("Saving fig")
            fig.savefig(f"figures_check/phase-diagram_{simname}.png")
        #---

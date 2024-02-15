import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
sys.path.append("/glade/u/home/tomasc/repos/xanimations")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from aux01_physfuncs import calculate_filtered_PV
from aux02_plotting import manual_facetgrid, get_orientation, BuRd
from cmocean import cm
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9
π = np.pi

if __name__ == "__main__": print("Starting hvid00 script")

#+++ MASTER DICTIONARY OF OPTIONS
plot_kwargs_by_var = {"u"         : dict(vmin=-0.01, vmax=+0.01, cmap=plt.cm.RdBu_r),
                      "PV_norm"   : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "PVᵍ_norm"  : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̃_norm"    : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "Ri"        : dict(vmin=-2, vmax=2, cmap=cm.balance),
                      "Ro"        : dict(vmin=-3, vmax=3, cmap=BuRd),
                      "εₖ"        : dict(norm=LogNorm(vmin=1e-10,   vmax=1e-8,   clip=True), cmap="inferno"),
                      "εₚ"        : dict(norm=LogNorm(vmin=1e-10/5, vmax=1e-8/5, clip=True), cmap="inferno"),
                      "ε̄ₖ"        : dict(norm=LogNorm(vmin=2e-11,   vmax=2e-9,   clip=True), cmap="inferno"),
                      "ε̄ₚ"        : dict(norm=LogNorm(vmin=2e-11/5, vmax=2e-9/5, clip=True), cmap="inferno"),
                      "Lo"        : dict(vmin=0, vmax=2, cmap=cm.balance),
                      "Δz_norm"   : dict(vmin=0, vmax=2, cmap=cm.balance),
                      "v"         : dict(vmin=-1.2 * 0.01, vmax=1.2 * 0.01, cmap=cm.balance),
                      "wb"        : dict(vmin=-1e-8, vmax=+1e-8, cmap=cm.balance),
                      "Kb"        : dict(vmin=-1e-1, vmax=+1e-1, cmap=cm.balance),
                      "γ"         : dict(vmin=0, vmax=1, cmap="plasma"),
                      "Π"         : dict(cmap=cm.balance, vmin=-1e-9, vmax=+1e-9),
                      "Πv"        : dict(cmap=cm.balance, vmin=-1e-9, vmax=+1e-9),
                      "Πh"        : dict(cmap=cm.balance, vmin=-1e-9, vmax=+1e-9),
                      "R_Π"       : dict(cmap=cm.balance, robust=True),
                      "Rᵍ_PVvs"   : dict(cmap=BuRd, vmin=0, vmax=1),
                      "R_PVvs"    : dict(cmap=BuRd, vmin=0, vmax=1),
                      "R_SPv"     : dict(cmap=BuRd, vmin=0, vmax=1),
                      "R_SPv2"    : dict(cmap=cm.balance, vmin=0, vmax=1),
                      "R_SPh"     : dict(cmap=cm.balance, vmin=0, vmax=1),
                      }
#---

#+++ Dynamic options
#+++ Figure out how this script is being run
try:
    shell = get_ipython().__class__.__name__
except NameError:
    shell = None
#---

if shell is not None:
    #+++ Running from IPython
    animate = False
    test = False
    time_avg = False
    summarize = True
    plotting_time = 19

    slice_names = ["tafields",]
    slice_names = ["xyi",]
    modifiers = ["-f2", "-S-f2", "", "-S"]
    modifiers = ["-f4",]

    varnames = ["q̃_norm"]
    contour_variable_name = None #"water_mask_buffered"
    contour_kwargs = dict(colors="y", linewidths=0.8, linestyles="--", levels=[0])
    #---
else:
    #+++ Running from python (probably from run_postproc.sh)
    animate = False
    test = False
    time_avg = False
    summarize = True

    slice_names = ["xyi", "xiz", "iyz", "tafields"]
    slice_names = ["tafields",]
    modifiers = ["",]

    varnames = ["εₖ", "PV_norm", "Ro"]
    varnames = ["ε̄ₖ"]
    contour_variable_name = None #"q̃_norm"
    contour_kwargs = dict(colors="y", linewidths=0.8, linestyles="--", levels=[0])
    #---

plot_kwargs_by_var = { k : plot_kwargs_by_var[k] for k in plot_kwargs_by_var if k in varnames}
#---

for modifier in modifiers:
    for slice_name in slice_names:

        #+++ Read and reorganize Dataset
        ncname = f"data_post/{slice_name}_snaps{modifier}.nc"
        if __name__ == "__main__": print(f"\nOpening {ncname}")
        snaps = xr.open_dataset(ncname)
        if not animate:
            snaps = snaps.sel(time=[plotting_time], method="nearest")

        if summarize:
            summary_values = [0.2, 1.25]
            snaps = snaps.sel(Ro_h = summary_values, Fr_h = summary_values)

        if "buffer" in snaps.coords:
            snaps = snaps.isel(buffer=-1)

        snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
        snaps = snaps.chunk(time=1)
        snaps["land_mask"] = snaps.land_mask.isel(Ro_h=0, Fr_h=0)
        snaps.PV_norm.attrs = dict(long_name=r"Normalized Ertel PV")
        #---

        #+++ Deal with dimensions and time-averaging
        if time_avg:
            print("time-averaging results")
            snaps = snaps.chunk(time="auto").sel(time=slice(15, None, 1)).mean("time", keep_attrs=True, keepdims=True)
            snaps = snaps.assign_coords(time=[0])

        try:
            snaps = snaps.reset_coords(("zC", "zF"))
            snaps = snaps.isel(λ=0)
        except ValueError:
            pass
        #---

        #+++ Adjust/create variables
        snaps.PV_norm.attrs = dict(long_name=r"Normalized Ertel PV")

        snaps["Lo"] = 2*π * np.sqrt(snaps["εₖ"]  / snaps["N²∞"]**(3/2))
        snaps["Δz_norm"] = snaps.Δz_min / snaps["Lo"]
        snaps["Δz_norm"].attrs = dict(long_name="Δz / Ozmidov scale")

        if ("Ro" in snaps.variables.keys()) and ("Ri" in snaps.variables.keys()):
            snaps["PVᵍ_norm"] = (1 + snaps.Ro - 1/snaps.Ri)

            snaps["Rᵍ_PVvs"] = (-snaps.Ri**(-1) / (1 + snaps.Ro - 1/snaps.Ri))#.where(snaps.CSI_mask)

        if "q̃" in snaps.variables.keys():
            snaps["q̃_norm"] = snaps["q̃"]  / (snaps["N²∞"] * snaps["f₀"])
            snaps["q̃_norm"].attrs = dict(long_name=r"Normalized filtered Ertel PV")

        if "wb" in snaps.variables.keys():
            snaps["Kb"] = -snaps.wb / snaps["N²∞"]

        snaps["γ"] = snaps["εₚ"] / (snaps["εₚ"] + snaps["εₖ"])

        if "Π" in snaps.variables.keys():
            Π_thres = 1e-11
            Π_std = snaps["Π"].where(snaps.water_mask).pnstd(("x", "y"))
            snaps["Πᵃ"] = snaps["Π"] - snaps["Πᵍ"]
            snaps["R_Π"]  = (snaps["Πᵃ"] / Π_std)#.where((snaps["Πᵃ"] > 0)          & (snaps["Π"] > Π_thres))

            q̃_vs = snaps["q̃ᵢ"].sel(i=[1,2]).sum("i")
            snaps["R_PVvs"] = (q̃_vs / snaps["q̃"])

            SP_v = snaps.SP.sel(j=3)
            snaps["R_SPv"]  = (SP_v / snaps.Π)#.where(snaps.CSI_mask)
            snaps["R_SPv2"] = (SP_v / snaps.Π)#.where(np.logical_and(snaps.CSI_mask, snaps.Π > 0))

            snaps["Πv"] = snaps.SP.sel(j=3)
            snaps["Πh"] = snaps.SP.sel(j=[1,2]).sum("j")
        #---

        #+++ Options
        sel = dict()
        if "xC" in snaps.coords: # has an x dimension
            sel = sel | dict(x=slice(-snaps.headland_intrusion_size_max/3, np.inf))
        if "yC" in snaps.coords: # has a y dimension
            sel = sel | dict(y=slice(snaps.yC[0] + snaps.sponge_length_y, np.inf))
        if "zC" in snaps.coords: # has a z dimension
            sel = sel | dict(z=slice(None))

        cbar_kwargs = dict(shrink=0.5, fraction=0.012, pad=0.02, aspect=30)
        if ("xC" in snaps.coords) and ("yC" in snaps.coords):
            figsize = (8, 8)
            cbar_kwargs = dict(location="right") | cbar_kwargs
        else:
            figsize = (9, 6)
            cbar_kwargs = dict(location="bottom") | cbar_kwargs

        opts_orientation = get_orientation(snaps)
        #---

        #+++ Begin plotting
        varlist = list(plot_kwargs_by_var.keys())
        for var in varlist:
            if var not in snaps.variables.keys():
                if __name__ == '__main__': print(f"Skipping {slice_name} slices of {var} since they don't seem to be in the file.")
                continue

            plot_kwargs = plot_kwargs_by_var[var]
            if contour_variable_name in snaps.variables.keys():
                contour_variable = snaps[contour_variable_name].pnsel(**sel)
            else:
                contour_variable = None
            kwargs = dict(plot_kwargs=(plot_kwargs | opts_orientation),
                          land_mask=snaps.land_mask.biject(),
                          contour_variable = contour_variable,
                          contour_kwargs = (contour_kwargs | opts_orientation),
                          add_abc = True,
                          label_Slope_Bu = True,)

            if animate:
                #+++ Animate!
                from xanimations import Movie

                if "time" not in snaps[var].coords.keys():
                    if __name__ == "__main__": print(f"Skipping {slice_name} slices of {var} for animating since they don't have a time dimension.")
                    continue

                anim_horvort = Movie(snaps[var].pnsel(**sel), plotfunc=manual_facetgrid,
                                     pixelwidth  = 1000 if (summarize and slice_name in ["xyi", "tafields"]) else 1800,
                                     pixelheight = 1000 if (summarize and slice_name in ["xyi", "tafields"]) else 1000,
                                     dpi=200,
                                     frame_pattern='frame_%05d.png',
                                     fieldname=None,
                                     input_check=False,
                                     **kwargs
                             )

                # The condition below is necessary for processes scheduler:
                # https://stackoverflow.com/questions/18204782/runtimeerror-on-windows-trying-python-multiprocessing
                if __name__ == '__main__':
                    print(f"Plotting var {var} on slice {slice_name} with modifier {modifier}")
                    from dask.diagnostics import ProgressBar

                    if summarize:
                        outname = f"anims/{var}_{slice_name}_summary_facetgrid{modifier}.mp4"
                    else:
                        outname = f"anims/{var}_{slice_name}_facetgrid{modifier}.mp4"
                    print("Start frame saving")
                    with ProgressBar():
                        anim_horvort.save(outname,
                                          ffmpeg_call = "/glade/u/home/tomasc/miniconda3/envs/py310/bin/ffmpeg",
                                          remove_frames=True,
                                          remove_movie=False,
                                          progress=True,
                                          verbose=False,
                                          overwrite_existing=True,
                                          framerate=12,
                                          parallel=True,
                                          #parallel_compute_kwargs=dict(num_workers=18, memory_limit='5GB', processes=False), # 24 min
                                          parallel_compute_kwargs=dict(num_workers=18, memory_limit='5GB', scheduler="processes"), # 2 min
                                          )
                    plt.close("all")
                #---

            else:
                #+++ Plot figure
                fig = plt.figure(figsize=figsize)
                axes, fig = manual_facetgrid(snaps[var].pnsel(**sel), fig, -1, **kwargs)

                if test:
                    print("Plotting snapshots for testing")
                    plt.show()
                    pause

                else:
                    if summarize:
                        outname = f"figures_check/{var}_{slice_name}_summary_facetgrid{modifier}.pdf"
                    else:
                        outname = f"figures_check/{var}_{slice_name}_facetgrid{modifier}.pdf"
                    print(f"Saving snapshots to {outname}")
                    fig.suptitle("")
                    fig.savefig(outname, dpi=200)
                #---
        #---

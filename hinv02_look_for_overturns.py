import sys
sys.path.append("/glade/u/home/tomasc/repos/xanimations")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux00_utils import open_simulation
from aux02_plotting import letterize, plot_kwargs_by_var

modifiers = ["", "-S"]
modifiers = ["",]
variables = ["PV_norm", "εₖ"]
Fr_h = 0.2
Ro_h = 1
animate = True

#+++ Pick downstream distances
if (Fr_h==0.08) and (Ro_h==1):
    downstream_distances = [0, 100, 200,]
elif (Fr_h==0.08) and (Ro_h==0.08):
    downstream_distances = [0, 50, 100,]
elif (Fr_h==0.2) and (Ro_h==0.2):
    downstream_distances = [0, 50, 100,]
elif (Fr_h==0.2) and (Ro_h==0.5):
    downstream_distances = [0, 75, 150,]
elif (Fr_h==0.2) and (Ro_h==1):
    downstream_distances = [0, 100, 200,]
else:
    downstream_distances = [0, 50, 100,]
#---

#+++ Animation function
def create_animation(ds_xz, fig, tt=None,
                     framedim="time",
                     variables = ["PV_norm"],
                     plot_kwargs = dict()):

    #+++ Create axes
    axes = fig.subplots(ncols=len(ds_xz.yC), nrows=len(variables), sharex=True, sharey=True)
    #---

    #+++ Plot stuff
    ds_xz = ds_xz.isel(time=tt)
    pcms = []
    for i, yC in enumerate(ds_xz.yC.values):
        for j, variable in enumerate(variables):
            ax = axes[j, i]
            pcm = ds_xz[variable].sel(yC=yC).pnplot(ax=ax, x="x", **plot_kwargs_by_var[variable], add_colorbar=False, rasterized=True)
            pcms.append(pcm)
            if i != 0:
                ax.set_title("")
    #---

    letterize(np.array([*axes.flatten(), ax]), x=0.05, y=0.9)
    return axes, fig
#---


for modifier in modifiers:
    if __name__ == "__main__": print(f"Opening modifier={modifier}")

    #+++ Open dataset
    path = f"./headland_simulations/data/"
    simname = f"NPN-R%sF%s{modifier}" % (str(Ro_h).replace(".", ""), str(Fr_h).replace(".", ""))
    grid_xyz, xyz = open_simulation(path + f"xyz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology="NPN",
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time=1)),
                                    )
    #---

    #+++ Adjust attributes
    xyz["land_mask"] = xyz["Δxᶜᶜᶜ"].where(xyz["Δxᶜᶜᶜ"] == 0)
    xyz["PV_norm"] = xyz.PV / (xyz.N2_inf * xyz.f_0)

    xyz = xyz.sel(xC=slice(-50, None), yC=slice(-250, 750))
    ds_xz = xyz.sel(yC=downstream_distances, method="nearest")

    ds_xz.PV_norm.attrs = dict(long_name=r"Ertel PV / $N^2_\infty f_0$")
    ds_xz["εₖ"].attrs = dict(long_name=r"$\varepsilon_k$ [m²/s³]")
    ds_xz["ω_y"].attrs = dict(long_name=r"$y$-vorticity [1/s]")
    ds_xz.xC.attrs["long_name"] = "$x$"
    ds_xz.zC.attrs["long_name"] = "$z$"
    #---

    #+++ Create figure
    size = 2.5
    fig = plt.figure(figsize=(15, 5))
    #---

    if animate:
        #+++ Animate
        from xanimations import Movie
        anim = Movie(ds_xz, plotfunc = create_animation,
                     pixelwidth  = 2000,
                     pixelheight = 900,
                     dpi = 200,
                     frame_pattern = "frame_%05d.png",
                     fieldname = None,
                     input_check = False,
                     variables = variables,
                     )

        from dask.diagnostics import ProgressBar
        with ProgressBar():
            outname = f"anims/{variables[0]}_progression_{simname}.mp4"
            anim.save(outname,
                      ffmpeg_call = "/glade/u/home/tomasc/miniconda3/envs/py310/bin/ffmpeg",
                      remove_frames = True,
                      remove_movie = False,
                      progress = True,
                      verbose = False,
                      overwrite_existing = True,
                      framerate = 8,
                      parallel = True,
                      #parallel_compute_kwargs=dict(num_workers=10, memory_limit="2GB"), # 2 min
                      #parallel_compute_kwargs=dict(num_workers=10, memory_limit="2GB", scheduler="processes"), # 2 min
                      )
            plt.close("all")
        #---

    else:
        fig, axes = create_animation(ds_xz, fig, tt=-1, framedim="time", variables=variables)


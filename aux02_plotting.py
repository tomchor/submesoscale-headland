import numpy as np
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm

#+++ Manual FacetGrid plot
def manual_facetgrid(da, fig, tt=None,
                     framedim="time",
                     plot_kwargs = dict(),
                     contour_variable = None,
                     contour_kwargs = dict(),
                     land_mask = None,
                     opts_land = dict(cmap="Set2", vmin=0, vmax=1, alpha=1.0, zorder=10,),
                     cbar_kwargs = dict(shrink=0.5, fraction=0.012, pad=0.02, aspect=30, location="right", orientation="vertical"),
                     bbox = dict(boxstyle="round", facecolor="white", alpha=0.8),
                     add_title = True,
                     add_abc = True,
                     label_Slope_Bu =False):
    """ Plot `da` as a FacetGrid plot """
    plt.rcParams["font.size"] = 9

    #+++ Create axes
    len_Fr = len(da.Fr_h)
    len_Ro = len(da.Ro_h)
    axes = fig.subplots(ncols=len_Fr, nrows=len_Ro,
                        sharex=True, sharey=True,)
    #---

    #+++ Get correct time
    if tt is not None:
        da = da.isel(time=tt, missing_dims="warn")
        if (contour_variable is not None) and ("time" in contour_variable.coords):
            contour_variable = contour_variable.isel(time=tt)

    if add_title:
        if "time" in da.coords.keys():
            fig.suptitle(f"Time = {da.time.item():.3g} advective times")
        else:
            fig.suptitle(f"Time-averaged")
    #---

    #+++ Plot each panel
    from string import ascii_lowercase
    alphabet = list(ascii_lowercase)
    for i_Ro, axs_Ro in enumerate(axes):
        Ro_h = da.Ro_h[i_Ro].item()
        for j_Fr, ax in enumerate(axs_Ro):
            Fr_h = da.Fr_h[j_Fr].item()
            im = da.sel(Fr_h=Fr_h, Ro_h=Ro_h).pnplot(ax=ax, add_colorbar=False, rasterized=True, **plot_kwargs)
            if contour_variable is not None:
                ct = contour_variable.sel(Fr_h=Fr_h, Ro_h=Ro_h).pncontour(ax=ax, add_colorbar=False, zorder=5, **contour_kwargs)

            if land_mask is not None:
                ax.pcolormesh(land_mask[land_mask.dims[-1]], land_mask[land_mask.dims[0]], land_mask.where(land_mask==1), rasterized=True, **opts_land)

            ax.set_title("")

            if i_Ro == 0:
                ax.set_title(f"$Fr_h =$ {Fr_h:.3g}", fontsize=9)
            if i_Ro != (len_Ro-1):
                ax.set_xlabel("")

            if j_Fr == (len_Fr-1):
                ax2 = ax.twinx()
                ax2.set_ylabel(f"$Ro_h =$ {Ro_h:.3g}", fontsize=9)
                ax2.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
                ax2.spines["top"].set_visible(False)
            if j_Fr != 0:
                ax.set_ylabel("")

            if add_abc:
                if label_Slope_Bu:
                    S_h = Ro_h/Fr_h
                    ax.text(0.05, 0.9, f"({alphabet.pop(0)})\n$S_h=${S_h:.3g}", transform=ax.transAxes, bbox=bbox, zorder=1e3, fontsize=7)
                else:
                    Bu_h = (Ro_h/Fr_h)**2
                    ax.text(0.05, 0.9, f"({alphabet.pop(0)})\n$Bu_h=${Bu_h:.3g}", transform=ax.transAxes, bbox=bbox, zorder=1e3, fontsize=7)
    #---
 
    if "label" not in cbar_kwargs.keys():
        label = da.long_name if "long_name" in da.attrs.keys() else da.longname if "longname" in da.attrs else da.name
        label += f" [{da.units}]" if "units" in da.attrs else ""
        fig.colorbar(im, ax=axes.ravel().tolist(), label=label, **cbar_kwargs)
    else:
        fig.colorbar(im, ax=axes.ravel().tolist(), **cbar_kwargs)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0, wspace=0)

    return axes, fig
#---

#+++ Get proper orientation for plotting
def get_orientation(ds):
    opts_orientation = dict()
    if "xC" in ds.coords: # has an x dimension
        opts_orientation = opts_orientation | dict(x="x")
    if "yC" in ds.coords: # has a y dimension
        opts_orientation = opts_orientation | dict(y="y")
    if "zC" in ds.coords: # has a z dimension
        opts_orientation = opts_orientation | dict(y="z")
    return opts_orientation
#---

#+++ Define seamount-plotting function
def fill_seamount_yz(ax, ds, color="silver"):
    from aux01_physfuncs import seamount_curve
    ax.fill_between(ds.yC, seamount_curve(ds.xC, ds.yC, ds), color=color)
    return

def fill_seamount_xy(ax, ds, radius, color="silver"):
    from matplotlib import pyplot as plt
    circle1 = plt.Circle((0, 0), radius=radius, color='silver', clip_on=False, fill=True)
    ax.add_patch(circle1)
    return
#---

#+++ Define mscatter, to plot scatter with a list of markers
def mscatter(x,y,ax=None, markers=None, **kw):
    """ Plots scatter but marker can be a list """
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (markers is not None) and (len(markers)==len(x)):
        paths = []
        for marker in markers:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc
#---

#+++ Get DataArray with proper colors and markers
def create_mc(bulk):
    """ Creates marker and color variables in `bulk` """
    import xarray as xr

    bulk["marker"] = xr.DataArray(len(bulk.Ro_h)*[len(bulk.Fr_h)*["o"]], dims=["Ro_h", "Fr_h"], coords=dict(Ro_h=bulk.Ro_h, Fr_h=bulk.Fr_h))
    bulk["color"] = xr.DataArray(len(bulk.Ro_h)*[len(bulk.Fr_h)*["black"]], dims=["Ro_h", "Fr_h"], coords=dict(Ro_h=bulk.Ro_h, Fr_h=bulk.Fr_h))

    # threed_sims
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==1.25) & (bulk.Fr_h==1.25)), other="^")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==1.25) & (bulk.Fr_h==1.25)), other="green")
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.5) & (bulk.Fr_h==0.5)), other="^")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.5) & (bulk.Fr_h==0.5)), other="green")

    # bathfo_sims
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.08) & (bulk.Fr_h==1.25)), other="X")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.08) & (bulk.Fr_h==1.25)), other="blue")
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.08) & (bulk.Fr_h==0.5)), other="X")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.08) & (bulk.Fr_h==0.5)), other="blue")
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.2) & (bulk.Fr_h==1.25)), other="X")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.2) & (bulk.Fr_h==1.25)), other="blue")

    # vertco_sims
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.08) & (bulk.Fr_h==0.08)), other="D")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.08) & (bulk.Fr_h==0.08)), other="orange")
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.2) & (bulk.Fr_h==0.2)), other="D")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.2) & (bulk.Fr_h==0.2)), other="orange")

    # vertsh_sims
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==0.5) & (bulk.Fr_h==0.08)), other="d")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==0.5) & (bulk.Fr_h==0.08)), other="orchid")
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==1.25) & (bulk.Fr_h==0.08)), other="d")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==1.25) & (bulk.Fr_h==0.08)), other="orchid")
    bulk["marker"] = bulk.marker.where(np.logical_not((bulk.Ro_h==1.25) & (bulk.Fr_h==0.2)), other="d")
    bulk["color"] = bulk.color.where(np.logical_not((bulk.Ro_h==1.25) & (bulk.Fr_h==0.2)), other="orchid")

    return bulk
#---

#+++ Instability angles
def plot_angles(angles, ax):
    vlim = ax.get_ylim()
    hlim = ax.get_xlim()
    length = np.max([np.diff(vlim), np.diff(hlim)])
    for α in np.array(angles):
        v0 = np.mean(vlim)
        h0 = np.mean(hlim)
        h1 = h0 + length*np.cos(α)
        v1 = v0 + length*np.sin(α)
        ax.plot([h0, h1], [v0, v1], c='k', ls='--')
    return
#---

#+++ Define colors and markers
color_base = ["b", "C1", "C2", "C3", "C5", "C8"]
marker_base = ["o", "v", "P"]

colors = color_base*len(marker_base)
markers = list(chain(*[ [m]*len(color_base) for m in marker_base ]))
#---

#+++ Standardized plotting
def plot_scatter(ds, ax=None, x=None, y=None, hue="simulation", add_guide=True, **kwargs):
    for i, s in enumerate(ds[hue].values):
        #++++ Getting values for specific point
        xval = ds.sel(**{hue:s})[x]
        yval = ds.sel(**{hue:s})[y]
        marker = markers[i]
        color = colors[i]
        #----

        #++++ Define label (or not)
        if add_guide:
            label=s
        else:
            label=""
        #----

        #++++ Plot
        ax.scatter(xval, yval, c=color, marker=marker, label=s, **kwargs)
        #----

        #++++ Include labels
        try:
            ax.set_xlabel(xval.attrs["long_name"])
        except:
            ax.set_xlabel(xval.name)

        try:
            ax.set_ylabel(yval.attrs["long_name"])
        except:
            ax.set_ylabel(yval.name)
        #----

    return 
#---

#+++ Letterize plot axes
def letterize(axes, x, y, coords=True, bbox=dict(boxstyle='round', 
                                                 facecolor='white', alpha=0.8),
                     **kwargs):
    from string import ascii_lowercase
    for ax, c in zip(axes.flatten(), ascii_lowercase*2):
        ax.text(x, y, c, transform=ax.transAxes, bbox=bbox, **kwargs)
    return
#---

#+++ Truncated colormaps
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap("RdBu_r")
BuRd = truncate_colormap(cmap, 0.05, 0.95)
#---

#+++ MASTER DICTIONARY OF OPTIONS
plot_kwargs_by_var = {"u"         : dict(vmin=-0.01, vmax=+0.01, cmap=plt.cm.RdBu_r),
                      "w"         : dict(vmin=-0.003, vmax=+0.003, cmap=plt.cm.RdBu_r),
                      "PV_norm"   : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "PVᶻ_norm"  : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "PVʰ_norm"  : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "PVᵍ_norm"  : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̃_norm"    : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̃ᶻ_norm"   : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̃ʰ_norm"   : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̄_norm"    : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̄"         : dict(vmin=-1e-11, vmax=1e-11, cmap="RdBu_r"),
                      "Ri"        : dict(vmin=-2, vmax=2, cmap=cm.balance),
                      "Ro"        : dict(vmin=-3, vmax=3, cmap=BuRd),
                      "εₖ"        : dict(norm=LogNorm(vmin=1e-10,   vmax=1e-8,   clip=True), cmap="inferno"),
                      "εₚ"        : dict(norm=LogNorm(vmin=1e-10/5, vmax=1e-8/5, clip=True), cmap="inferno"),
                      "ε̄ₖ"        : dict(norm=LogNorm(vmin=2e-11,   vmax=2e-9,   clip=True), cmap="inferno"),
                      "ε̄ₚ"        : dict(norm=LogNorm(vmin=2e-11/5, vmax=2e-9/5, clip=True), cmap="inferno"),
                      "Lo"        : dict(vmin=0, vmax=2, cmap=cm.balance),
                      "Δz_norm"   : dict(vmin=0, vmax=2, cmap=cm.balance),
                      "v"         : dict(vmin=-1.2 * 0.01, vmax=1.2 * 0.01, cmap=cm.balance),
                      "wb"        : dict(vmin=-1e-8, vmax=+1e-8, cmap=BuRd),
                      "w̄b̄"        : dict(vmin=-1e-8, vmax=+1e-8, cmap=BuRd),
                      "⟨w′b′⟩ₜ"   : dict(vmin=-1e-9, vmax=+1e-9, cmap=BuRd),
                      "uᵢGᵢ"      : dict(vmin=-1e-7, vmax=+1e-7, cmap=cm.balance),
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

label_dict = {"ε̄ₖ"     : r"Time-averaged KE dissipation rate $\bar\varepsilon_k$ [m²/s³]",
              "Ro"     : r"$Ro$ [vertical vorticity / $f_0$]",
              "q̃_norm" : r"Normalized filtered Ertel PV",
              "v"      : r"$v$-velocity [m/s]",
              }
#---


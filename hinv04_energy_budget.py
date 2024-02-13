import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
from aux00_utils import open_simulation
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from scipy.optimize import curve_fit

use_xyz = False
modifier = "-f4"
path = f"./headland_simulations/data/"
simname = f"NPN-R05F05{modifier}"
grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                use_advective_periods=True,
                                topology="NPN",
                                squeeze=True,
                                load=False,
                                open_dataset_kwargs=dict(chunks=dict(time="auto")),
                                )
grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                use_advective_periods=True,
                                topology="NPN",
                                squeeze=True,
                                load=False,
                                open_dataset_kwargs=dict(chunks=dict(time="auto")),
                                )

xyz = xyz.sel(time=[20, 30], method="nearest")
xyi = xyi.sel(time=slice(xyi.T_advective_spinup, None))

xyi0 = xyi.sel(time=xyz.time)

dxdz = (xyz["Δxᶜᶜᶜ"] * xyz["Δzᶜᶜᶜ"]).where((20 <= xyz.zC) & (xyz.zC <= 60))

terms = ["uᵢ∂ⱼuⱼuᵢ", "uᵢ∂ᵢp", "uᵢbᵢ", "uᵢ∂ⱼτᵢⱼ", "uᵢ∂ⱼτᵇᵢⱼ"]
terms_int = [ f"∫∫⁰{term}dxdz" for term in terms ]
for term, term_int in zip(terms, terms_int):
    xyz[term_int] = (xyz[term] * dxdz).pnsum(("x", "z"))

abs(xyi["∫∫⁰uᵢ∂ᵢpdxdz"]).pnplot(x="y", y="time", norm=LogNorm(vmin=1e-6, vmax=1e-2), cmap=plt.cm.inferno)

pause
xyi["∫∫⁰uᵢ∂ᵢpdxdz"].pnplot(x="y", hue="time")
xyz["∫∫⁰uᵢ∂ᵢpdxdz"].pnplot(x="y", hue="time", ls="--")



pause

opts = dict(vmin=-4e-8, vmax=4e-8, cmap=plt.cm.RdBu_r)
for term_int in terms_int:
    xyz[term_int].pnplot(x="y", label=term_int)
plt.legend()





pause

bulk = xr.load_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.sel(Ro_h=xyz.Ro_h, Fr_h=xyz.Fr_h, buffer=0, method="nearest")
bulk["∫∫ʷuᵢ∂ᵢpdxdz"].pnplot(x="y", ls="--")

bulk["net"] = -bulk["∫∫ʷuᵢ∂ⱼuⱼuᵢdxdz"] - bulk["∫∫ʷuᵢ∂ᵢpdxdz"] + bulk["∫∫ʷuᵢbᵢdxdz"] - bulk["∫∫ʷuᵢ∂ⱼτᵢⱼdxdz"] - bulk["∫∫ʷuᵢ∂ⱼτᵇᵢⱼdxdz"]
#bulk["net"] = -bulk["∫∫ʷuᵢ∂ⱼuⱼuᵢdxdz"] + bulk["∫∫ʷuᵢbᵢdxdz"] - bulk["∫∫ʷuᵢ∂ⱼτᵢⱼdxdz"] - bulk["∫∫ʷuᵢ∂ⱼτᵇᵢⱼdxdz"]

bulk.net.pnplot(x="y", label="dedt")
bulk["∫∫ʷεₖdxdz"].pnplot(x="y", label="dissipation")
bulk["∫∫ʷuᵢ∂ⱼτᵢⱼdxdz"].pnplot(x="y", label="full stress term")
plt.legend()

"""
<xarray.DataArray '∫∫ʷuᵢ∂ᵢpdxdz' ()>
array(6704.26099267)

<xarray.DataArray '∫∫ʷuᵢ∂ⱼτᵢⱼdxdz' ()>
array(1.52563076e-06)

"""

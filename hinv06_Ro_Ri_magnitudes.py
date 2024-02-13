import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
π = np.pi

resolution = ""
slice_name = "xyi"
#slice_name = "xiz"
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps_{resolution}.nc")
snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
#snaps = snaps.isel(time=-1)

np.sqrt((snaps.Ro.where(snaps.Ro<0)**2).pnmean(("time", "x", "y"))).pnplot(x="Ro_h", hue="Fr_h", marker="o")

#plt.figure()
#np.sqrt((snaps.Ri.where(snaps.Ro<0)**2).pnmean(("time", "x", "y"))).pnplot(hue="Ro_h", x="Fr_h", marker="o")

snaps["Fr"] = np.sign(snaps.Ri) / np.sqrt(abs(snaps.Ri))
plt.figure()
np.sqrt((snaps.Fr.where(snaps.Ro<0)**2).pnmean(("time", "x", "y"))).pnplot(hue="Ro_h", x="Fr_h", marker="o")
#np.sqrt((snaps.Fr.where(abs(snaps.Fr>0.5))**2).pnmean(("time", "x", "y"))).pnplot(hue="Ro_h", x="Fr_h", marker="o")

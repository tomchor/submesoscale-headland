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


#+++ Options for hvid00
animate = False
test = False
time_avg = False
summarize = True
zoom = True
plotting_time = 23
figdir = "figures"

slice_names = ["iyz",]
modifiers = ["-f2", "",]
modifiers = ["",]

varnames = ["v̂"]
contour_variable_name = None #"water_mask_buffered"
contour_kwargs = dict(colors="y", linewidths=0.8, linestyles="--", levels=[0])
#---

exec(open("hvid00_facetgrid.py").read())

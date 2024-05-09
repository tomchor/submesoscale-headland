import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

α_array = np.array([0.07, 0.1, 0.2])
V_array = np.array([0.5, 1]) # m
N_array = np.array([7e-3, 10e-3]) # 1/s

α = xr.DataArray(α_array, dims=["α"], coords=dict(α=α_array))
H = 400 # m; half the total depth, which is ≈ 800 m
L = H / α # m; half the intrusion length at the bottom (approx between 4 and 8 km for the bahamas)
V = xr.DataArray(V_array, dims=["V"], coords=dict(V=V_array))
N = xr.DataArray(N_array, dims=["N"], coords=dict(N=N_array))
f = 6.6e-5 # 1/s

Ro_h = V / (f * L)
Fr_h = V / (N * H)
S_h  = Ro_h / Fr_h

ΔLa = 28 - 25.5
ΔLo = 79.5 - 78

#+++ Gula et al.'s results
Δx_Gula = ΔLo * 100e3 # m
Δy_Gula = ΔLa * 111e3 # m
Δz_Gula = 800 # m
ΔV_Gula = Δx_Gula * Δy_Gula * Δz_Gula # m³

ε_scale_Gula = V**3 / L

bulk = xr.Dataset()
bulk["ρ∫∫∫ε̄ₖdxdydz_Gula"] = 0.5e9 # W
ρ = 1000
bulk["∫∫∫ε̄ₖdxdydz_Gula"] = bulk["ρ∫∫∫ε̄ₖdxdydz_Gula"] / ρ
bulk["⟨ε̄ₖ⟩_Gula"] = bulk["∫∫∫ε̄ₖdxdydz_Gula"] / ΔV_Gula
bulk["ε_norm_Gula"] = bulk["⟨ε̄ₖ⟩_Gula"] / ε_scale_Gula
#---

#+++
Δx_Chor = (800 + 400) # m
Δy_Chor = 3000 # m
Δz_Chor = 84 # m
ΔV_Chor = Δx_Chor * Δy_Chor * Δz_Chor

ε_scale_Chor = 0.01**3 / 400 # (m/s)³ / m

bulk["∫∫∫ε̄ₖdxdydz_Chor"] = 1e-2 # W m³ / kg
bulk["ρ∫∫∫ε̄ₖdxdydz_Chor"] = ρ * bulk["∫∫∫ε̄ₖdxdydz_Chor"]
bulk["⟨ε̄ₖ⟩_Chor"] = bulk["∫∫∫ε̄ₖdxdydz_Chor"] / ΔV_Chor
bulk["ε_norm_Chor"] = bulk["⟨ε̄ₖ⟩_Chor"] / ε_scale_Chor
# 1 Watt = kg m² / s³

print("Gula et al.'s average dissipation: ", bulk["⟨ε̄ₖ⟩_Gula"])
print("Chor & Wenegrat's average dissipation: ", bulk["⟨ε̄ₖ⟩_Chor"])
print()
print("Gula et al.'s normalized average dissipation: ", bulk["ε_norm_Gula"])
print("Chor & Wenegrat's normalized average dissipation: ", bulk["ε_norm_Chor"])

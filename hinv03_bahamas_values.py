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
Sb_h  = Ro_h / Fr_h

ΔLa = 28 - 25.5
ΔLo = 79.5 - 78

#+++ Gula et al.'s results
Δx_Gula = ΔLo * 100e3 # m
Δy_Gula = ΔLa * 111e3 # m
Δz_Gula = 800 # m
ΔV_Gula = Δx_Gula * Δy_Gula * Δz_Gula # m³

ε_scale_Gula = V**3 / L

norm = xr.Dataset()

norm["εₖ_max_Gula"] = 1e-5
norm["εₖ_max_norm_Gula"] = norm["εₖ_max_Gula"] / ε_scale_Gula

# Reminder: 1 Watt = kg m² / s³
norm["ρ∫∫∫ε̄ₖdxdydz_Gula"] = 0.5e9 # W
ρ = 1000
norm["∫∫∫ε̄ₖdxdydz_Gula"] = norm["ρ∫∫∫ε̄ₖdxdydz_Gula"] / ρ
norm["⟨ε̄ₖ⟩_Gula"] = norm["∫∫∫ε̄ₖdxdydz_Gula"] / ΔV_Gula
norm["εₖ_norm_Gula"] = norm["⟨ε̄ₖ⟩_Gula"] / ε_scale_Gula
#---

#+++ Chor and Wenegrat's results
Δx_Chor = (800 + 400) # m
Δy_Chor = 3000 # m
Δz_Chor = 84 # m
ΔV_Chor = Δx_Chor * Δy_Chor * Δz_Chor

ε_scale_Chor = 0.01**3 / 400 # (m/s)³ / m

norm["εₖ_max_Chor"] = 1e-8
norm["εₖ_max_norm_Chor"] = norm["εₖ_max_Chor"] / ε_scale_Chor

norm["∫∫∫ε̄ₖdxdydz_Chor"] = 1e-2 # W m³ / kg
norm["ρ∫∫∫ε̄ₖdxdydz_Chor"] = ρ * norm["∫∫∫ε̄ₖdxdydz_Chor"]
norm["⟨ε̄ₖ⟩_Chor"] = norm["∫∫∫ε̄ₖdxdydz_Chor"] / ΔV_Chor
norm["εₖ_norm_Chor"] = norm["⟨ε̄ₖ⟩_Chor"] / ε_scale_Chor
#---

#+++ Print results
α_value = 0.07
N_value = 0.01
norm = norm.sel(α=α_value)
Ro_h = Ro_h.sel(α=α_value)
Fr_h = Fr_h.sel(N=N_value)
Sb_h = Sb_h.sel(α=α_value, N=N_value)
for V in norm.V.values:
    print(f"\n---- For V∞ = {V} m/s ----\n")
    print(f"Gula's Roₕ = ", Ro_h.sel(V=V).item())
    print(f"Gula's Frₕ = ", Fr_h.sel(V=V).item())
    print(f"Gula's Sbₕ = ", Sb_h.sel(V=V).item())
    print()

    print("Gula et al.'s normalized instantaneous dissipation: ", norm["εₖ_max_norm_Gula"].sel(V=V).item())
    print("Chor & Wenegrat's normalized instantaneous dissipation: ", norm["εₖ_max_norm_Chor"].item())
    print()
    print("Gula et al.'s normalized average dissipation: ", norm["εₖ_norm_Gula"].sel(V=V).item(), "m²/s³")
    print("Chor & Wenegrat's normalized average dissipation: ", norm["εₖ_norm_Chor"].item(), "m²/s³")
    print()

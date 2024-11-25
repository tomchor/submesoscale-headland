import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
g = 9.81 # m/s²
ρ0 = 1000 # kg/m³
# Reminder: 1 Watt = kg m² / s³

#+++ Gula et al.'s results
α_array_Gula = np.array([0.07, 0.1, 0.2])
V_array_Gula = np.array([0.5, 1]) # m
N_array_Gula = np.array([7e-3, 10e-3]) # 1/s

α_Gula = xr.DataArray(α_array_Gula, dims=["α"], coords=dict(α=α_array_Gula))
H_Gula = 400 # m; half the total depth, which is ≈ 800 m
L_Gula = H_Gula / α_Gula # m; half the intrusion length at the bottom (approx between 4 and 8 km for the bahamas)
V_Gula = xr.DataArray(V_array_Gula, dims=["V"], coords=dict(V=V_array_Gula))
N_Gula = xr.DataArray(N_array_Gula, dims=["N"], coords=dict(N=N_array_Gula))
f_Gula = 6.6e-5 # 1/s

Gula = xr.Dataset()

Gula["Ro_h"] = V_Gula / (f_Gula * L_Gula)
Gula["Fr_h"] = V_Gula / (N_Gula * H_Gula)
Gula["Sb_h"] = Gula.Ro_h / Gula.Fr_h

ΔLa = 28 - 25.5
ΔLo = 79.5 - 78

Δx_Gula = ΔLo * 100e3 # m
Δy_Gula = ΔLa * 111e3 # m
Δz_Gula = 800 # m
ΔV_Gula = Δx_Gula * Δy_Gula * Δz_Gula # m³
LLH_Gula = L_Gula * L_Gula * H_Gula

Gula["ε_scale"] = V_Gula**3 / L_Gula

Gula["εₖ_max"] = 1e-5
Gula["εₖ_max_norm"] = Gula["εₖ_max"] / Gula.ε_scale

Gula["ρ∫∫∫ε̄ₖdxdydz"] = 0.5e9 # W
ρ = 1000
Gula["∫∫∫ε̄ₖdxdydz"] = Gula["ρ∫∫∫ε̄ₖdxdydz"] / ρ
Gula["⟨ε̄ₖ⟩"] = Gula["∫∫∫ε̄ₖdxdydz"] / LLH_Gula
Gula["εₖ_avg_norm"] = Gula["⟨ε̄ₖ⟩"] / Gula.ε_scale
#---

#+++ Srinivasan et alia (2021)'s results
α_array_Sri21 = np.array([0.05])
V_array_Sri21 = np.array([0.1,]) # m
N_array_Sri21 = np.array([6e-3,]) # 1/s

α_Sri21 = xr.DataArray(α_array_Sri21, dims=["α"], coords=dict(α=α_array_Sri21))
H_Sri21 = 400 # m; half the total depth, which is ≈ 800 m
L_Sri21 = H_Sri21 / α_Sri21 # m; half the intrusion length at the bottom (approx between 4 and 8 km for the bahamas)
V_Sri21 = xr.DataArray(V_array_Sri21, dims=["V"], coords=dict(V=V_array_Sri21))
N_Sri21 = xr.DataArray(N_array_Sri21, dims=["N"], coords=dict(N=N_array_Sri21))
f_Sri21 = 1.25e-5 # 1/s

Sri21 = xr.Dataset()

Sri21["Ro_h"] = V_Sri21 / (f_Sri21 * L_Sri21)
Sri21["Fr_h"] = V_Sri21 / (N_Sri21 * H_Sri21)
Sri21["Sb_h"] = Sri21.Ro_h / Sri21.Fr_h

Sri21["ε_scale"] = V_Sri21**3 / L_Sri21

Sri21["εₖ_max"] = 1e-9
Sri21["εₖ_max_norm"] = Sri21["εₖ_max"] / Sri21.ε_scale
#---

#+++ Nagai et alia's results
α_array_Nagai = np.array([0.05, 0.1, 0.2])
V_array_Nagai = np.array([0.5, 1]) # m

Δρ1 = 26.0 - 24.5 # kg/m³
Δh1 = 500 # m
Δρ2 = 25.5 - 24.5 # kg/m³
Δh2 = 100 # m
N1 = np.sqrt((g / ρ0) * Δρ1 / Δh1)
N2 = np.sqrt((g / ρ0) * Δρ2 / Δh2)
N_array_Nagai = np.array([N1, N2]) # 1/s

α_Nagai = xr.DataArray(α_array_Nagai, dims=["α"], coords=dict(α=α_array_Nagai))
H_Nagai = 200 # m; half the total depth, which is ≈ 400 m
L_Nagai = H_Nagai / α_Nagai # m; half the intrusion length at the bottom (approx between 2 and 5 km for the kuroshio islands)
V_Nagai = xr.DataArray(V_array_Nagai, dims=["V"], coords=dict(V=V_array_Nagai))
N_Nagai = xr.DataArray(N_array_Nagai, dims=["N"], coords=dict(N=N_array_Nagai))
f_Nagai = 7.3e-5 # 1/s

Nagai = xr.Dataset()

Nagai["Ro_h"] = V_Nagai / (f_Nagai * L_Nagai)
Nagai["Fr_h"] = V_Nagai / (N_Nagai * H_Nagai)
Nagai["Sb_h"] = Nagai.Ro_h / Nagai.Fr_h

Nagai["ε_scale"] = V_Nagai**3 / L_Nagai

Nagai["εₖ_max"] = 10**(-6.5)
Nagai["εₖ_max_norm"] = Nagai["εₖ_max"] / Nagai.ε_scale
#---

#+++ Chor and Wenegrat's results
L_Chor = 200 # m
H_Chor = 40 # m
V_Chor = 0.01 # m

Δx_Chor = (800 + 400) # m
Δy_Chor = 3000 # m
Δz_Chor = 84 # m
ΔV_Chor = Δx_Chor * Δy_Chor * Δz_Chor
LLH_Chor = L_Chor * L_Chor * H_Chor


Chor = xr.Dataset()

Chor["ε_scale"] = V_Chor**3 / L_Chor # (m/s)³ / m

Chor["εₖ_max"] = 1e-9 # m²/s³
Chor["εₖ_max_norm"] = Chor["εₖ_max"] / Chor.ε_scale
Chor["εₖ_avg_norm"] = 8e-1
#---

#+++ Print results
N_value = 0.01 # 1/s
V_value_Gula = 1 # m/s
α_value_Gula = 0.07
Gula = Gula.sel(α=α_value_Gula, V=V_value_Gula, N=N_value)

α_value_Nagai = 0.1
V_value_Nagai = 0.5 # m/s
Nagai = Nagai.sel(α=α_value_Nagai, V=V_value_Nagai, N=N_value, method="nearest")

print(f"Gula's Roₕ = ", Gula.Ro_h.item())
print(f"Gula's Frₕ = ", Gula.Fr_h.item())
print(f"Gula's Sbₕ = ", Gula.Sb_h.item())
print()
print(f"Srinivasan (2021)'s Roₕ = ", Sri21.Ro_h.item())
print(f"Srinivasan (2021)'s Frₕ = ", Sri21.Fr_h.item())
print(f"Srinivasan (2021)'s Sbₕ = ", Sri21.Sb_h.item())
print()
print(f"Nagai's Roₕ = ", Nagai.Ro_h.item())
print(f"Nagai's Frₕ = ", Nagai.Fr_h.item())
print(f"Nagai's Sbₕ = ", Nagai.Sb_h.item())
print()

print("Srinivasan Gula et alia (2021)'s normalized instantaneous dissipation: ", Sri21["εₖ_max_norm"].item())
print("Nagai et al.'s normalized instantaneous dissipation: ", Nagai["εₖ_max_norm"].item())
print("Gula et al.'s normalized instantaneous dissipation: ", Gula["εₖ_max_norm"].item())
print("Chor & Wenegrat's normalized instantaneous dissipation: ", Chor["εₖ_max_norm"].item())
print()
print("Gula et al.'s normalized average dissipation: ", Gula["εₖ_avg_norm"].item())
print("Chor & Wenegrat's normalized average dissipation: ", Chor["εₖ_avg_norm"].item())
print()

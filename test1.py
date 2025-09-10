# simulation_hotspot_structure.py
# Script minimal : profil T(r) gaussien -> radiance Planck -> mélange pixel -> T_eff -> plots
# Dépendances : numpy, matplotlib, scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C

# ----- fonctions utiles -----
h = C.h
c = C.c
kB = C.k
sigma = C.Stefan_Boltzmann

def planck_lambda_per_um(wl_m, T):
    """
    Radiance spectrale de Planck par unité de longueur d'onde,
    renvoie W·m^-2·sr^-1·um^-1
    wl_m : 1D array (m)
    T : scalar ou 1D array (K)
    """
    wl = np.asarray(wl_m, dtype=float)[:, None]         # (n_wl, 1)
    T_arr = np.atleast_1d(T)[None, :]                  # (1, nT)
    x = (h * c) / (wl * kB * T_arr)                    # (n_wl, nT)
    num = 2.0 * h * c**2
    den = (wl**5) * np.expm1(x)
    B_per_m = num / den                                # W·m^-2·sr^-1·m^-1
    B_per_um = B_per_m * 1e-6                          # W·m^-2·sr^-1·um^-1
    if B_per_um.shape[1] == 1:
        return B_per_um[:, 0]                          # (n_wl,)
    return B_per_um                                    # (n_wl, nT)

# ----- paramètres (modifiable) -----
Tmax = 1600.0           # K (centre du hotspot)
Tmin = 130.0            # K (fond nuit)
d_MAJIS = 77e3          # m (77 km/px) -> mettre 57e3 pour jour si besoin
r0_list_km = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100]  # exemples de r0 en km
r0_list = [r * 1e3 for r in r0_list_km]

# domaine spectral (µm)
wl_um = np.linspace(0.5, 5.5, 300)
wl_m = wl_um * 1e-6

# storage
L_pixel_list = []
T_eff_list = []
fill_list = []

# ----- boucle sur r0 -----
for r0 in r0_list:
    # grille radiale sur le hotspot
    nr = 200
    r = np.linspace(0.0, r0, nr)          # m
    dr = r[1] - r[0]
    dA_ring = 2.0 * np.pi * r * dr       # élément d'aire d'un anneau
    area_total = np.pi * r0**2

    # profil de température gaussien
    T_r = (Tmax - Tmin) * np.exp(- (r**2) / (r0**2)) + Tmin   # (nr,)

    # radiance du hotspot moyennée sur la surface (intégration radiale)
    B_wl_r = planck_lambda_per_um(wl_m, T_r)    # (n_wl, nr)
    numer = np.sum(B_wl_r * dA_ring[None, :], axis=1)   # (n_wl,)
    L_hotspot_avg = numer / area_total

    # radiance de fond (ici nuit = corps noir à Tmin)
    L_background = planck_lambda_per_um(wl_m, Tmin)     # (n_wl,)

    # fill factor (fraction de pixel occupée)
    f_fill = (np.pi * r0**2) / (d_MAJIS**2)
    f_fill = min(f_fill, 1.0)

    # radiance perçue par le pixel (mélange fond + hotspot)
    L_pixel = (1.0 - f_fill) * L_background + f_fill * L_hotspot_avg

    # conversion en T_eff :
    # 1) intégrer L_pixel(λ) sur λ -> W·m^-2·sr^-1
    # 2) multiplier par π -> exitance spectrale totale en W·m^-2
    integrated_radiance_sr = np.trapezoid(L_pixel, wl_um)    # W·m^-2·sr^-1
    exitance = np.pi * integrated_radiance_sr           # W·m^-2
    T_eff = (exitance / sigma) ** 0.25

    # stocker
    L_pixel_list.append(L_pixel)
    T_eff_list.append(float(T_eff))
    fill_list.append(f_fill)

# ----- affichage des résultats -----
print("r0 (km)  fill_factor     T_eff (K)")
for r0_km, f, Te in zip(r0_list_km, fill_list, T_eff_list):
    print(f"{r0_km:6.2f}    {f:12.3e}    {Te:8.2f}")

# plot : spectres
plt.figure(figsize=(8,5))
for r0_km, L in zip(r0_list_km, L_pixel_list):
    plt.plot(wl_um, L, label=f"r0={r0_km} km")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Radiance spectral (W·m$^{-2}$·sr$^{-1}$·µm$^{-1}$)")
plt.title("Spectres simulés du pixel")
plt.legend()
plt.grid(True)
plt.tight_layout()


# plot : profils T(r) avec axe radial en log
plt.figure(figsize=(8,5))
for r0_km in r0_list_km:
    r_km = np.logspace(-3, np.log10(5*r0_km), 300)  # de 1e-3 km (=1 m) jusqu’à r0
    T_r = (Tmax - Tmin) * np.exp(- ( (r_km*1e3)**2 ) / ((r0_km*1e3)**2)) + Tmin
    plt.plot(r_km, T_r, label=f"r0={r0_km} km")

plt.xscale("log")
plt.xlabel("Radius (km)")
plt.ylabel("Temperature (K)")
plt.title("Profils radiaux T(r)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
# fin de script

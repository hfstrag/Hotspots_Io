# simulation_hotspot_structure_um.py
# Tout en µm (longueurs d’onde)
# Script : profil T(r) gaussien -> radiance Planck -> mélange pixel -> T_eff -> plots
# Dépendances : numpy, matplotlib, scipy, pandas

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C
import pandas as pd

# ----- constantes -----
h = C.h
c = C.c
kB = C.k
sigma = C.Stefan_Boltzmann

# ----- fonctions utiles -----
def planck_lambda_per_um(wl_um, T):
    """
    Radiance spectrale de Planck par µm
    wl_um : array (µm)
    T : scalaire ou array (K)
    Retourne W·m^-2·sr^-1·µm^-1
    """
    wl_m = np.asarray(wl_um) * 1e-6              # m
    wl_m = wl_m[:, None]                         # (n_wl, 1)
    T_arr = np.atleast_1d(T)[None, :]            # (1, nT)
    x = (h * c) / (wl_m * kB * T_arr)
    num = 2.0 * h * c**2
    den = (wl_m**5) * np.expm1(x)
    B_per_m = num / den                          # W·m^-2·sr^-1·m^-1
    B_per_um = B_per_m * 1e-6                    # W·m^-2·sr^-1·µm^-1
    if B_per_um.shape[1] == 1:
        return B_per_um[:, 0]
    return B_per_um

def get_radiance(wvl_um, i=0.0, e=0.0, filename=None):
    """
    Charge un spectre de radiance depuis un CSV et l’interpole sur une grille en µm.
    CSV attendu : colonnes [Wavelength (nm), Radiance (W/m²/sr/µm)]
    """
    data = pd.read_csv(filename, header=0)
    data = data.rename(columns={
        "Wavelength (nm)": "wvl_nm",
        "Radiance (W/m²/sr/µm)": "rad"
    })
    wvl_file_um = data["wvl_nm"].to_numpy() / 1000.0   # nm -> µm
    rad_file = data["rad"].to_numpy()

    # interpolation
    rad_interp = np.interp(wvl_um, wvl_file_um, rad_file, left=0, right=0)

    # correction géométrique simple (Lambert)
    mu_i = np.cos(np.radians(i))
    mu_e = np.cos(np.radians(e))
    return rad_interp * mu_i * mu_e

# ----- paramètres -----
periode = "jour"   # "jour" ou "nuit"
Tmax = 1600.0      # K (centre du hotspot)
Tmin = 130.0       # K (fond)
d_MAJIS = 77e3     # m (77 km/px ; mettre 57e3 pour jour)
r0_list_km = [0.1, 0.2, 0.5, 1, 2.5, 5, 10, 25, 40]
r0_list = [r * 1e3 for r in r0_list_km]

# domaine spectral en µm
wl_um = np.linspace(0.5, 5.8, 300)

# storage
L_pixel_list = []
T_eff_list = []
fill_list = []

# radiance de fond (background)
if periode == "jour":
        wvl_all_um = wl_um  # µm
        io_rad_nims  = get_radiance(wvl_all_um, i=14.13, e=0, filename="Io_NIMS.csv")
        io_rad_jiram = get_radiance(wvl_all_um, i=14.13, e=0, filename="Io_JIRAM_JM0261.csv")

        # normalisation entre spectres
        mask = (wvl_all_um > 2.5) & (wvl_all_um < 5.0)
        norm_factor = np.mean(io_rad_jiram[mask]) / np.mean(io_rad_nims[mask])
        L_background = io_rad_nims * norm_factor

else:
        L_background = planck_lambda_per_um(wl_um, Tmin)

# ----- boucle sur r0 pour calcul de chaque T_eff -----
for r0 in r0_list:
    # grille radiale
    nr = 200
    r = np.linspace(0.0, r0, nr)    # m
    dr = r[1] - r[0]
    dA_ring = 2.0 * np.pi * r * dr
    area_total = np.pi * r0**2

    # profil T(r)
    T_r = (Tmax - Tmin) * np.exp(- (r**2) / (r0**2)) + Tmin

    # radiance hotspot moyenne
    B_wl_r = planck_lambda_per_um(wl_um, T_r)   # (n_wl, nr)
    numer = np.sum(B_wl_r * dA_ring[None, :], axis=1)
    L_hotspot_avg = numer / area_total

    # fill factor du hotspot dans le pixel
    f_fill = min((np.pi * r0**2) / (d_MAJIS**2), 1.0)

    # pondération entre la radiance du hotspot et la radiance du fond
    L_pixel = (1.0 - f_fill) * L_background + f_fill * L_hotspot_avg

    # T_eff
    integrated_radiance_sr = np.trapezoid(L_pixel, wl_um)   # W·m^-2·sr^-1
    exitance = np.pi * integrated_radiance_sr           # W·m^-2
    T_eff = (exitance / sigma) ** 0.25

    # stockage
    L_pixel_list.append(L_pixel)
    T_eff_list.append(float(T_eff))
    fill_list.append(f_fill)

# ----- affichage -----
print("r0 (km)  fill_factor     T_eff (K)")
for r0_km, f, Te in zip(r0_list_km, fill_list, T_eff_list):
    print(f"{r0_km:6.2f}    {f:12.3e}    {Te:8.2f}")

# Spectres
plt.figure(figsize=(8,5))

for r0_km, L in zip(r0_list_km, L_pixel_list):
    plt.semilogy(wl_um, L)
    plt.plot(wl_um, L, label=f"r0={r0_km} km")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Radiance (W·m$^{-2}$·sr$^{-1}$·µm$^{-1}$)")
plt.title("Spectres simulés du pixel")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Profils T(r)
plt.figure(figsize=(8,5))
for r0_km in r0_list_km:
    r_km = np.logspace(-3, np.log10(5*r0_km), 300)  # de 1e-3 km (=1 m) à 5*r0
    T_r = (Tmax - Tmin) * np.exp(- ((r_km*1e3)**2) / ((r0_km*1e3)**2)) + Tmin
    plt.plot(r_km, T_r, label=f"r0={r0_km} km")
plt.xscale("log")
plt.xlabel("Radius (km)")
plt.ylabel("Temperature (K)")
plt.title("Profils radiaux T(r)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()

# T_eff vs r0
plt.figure(figsize=(7,5))
plt.plot(r0_list_km, T_eff_list, "o-", lw=2)
plt.xlabel("Hotspot radius r0 (km)")
plt.ylabel("Effective temperature T_eff (K)")
plt.title("T_eff en fonction de la taille du hotspot")
plt.grid(True)
plt.tight_layout()

plt.show()
# fin de script
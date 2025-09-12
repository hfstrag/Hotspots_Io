#### simulation des hotspots volcaniques d’Io 
## Structure du script : 

# Dépendances : numpy, matplotlib, scipy, pandas

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C
import pandas as pd

# ----- constantes fondamentales ------ 
h = C.h
c = C.c
kB = C.k
sigma = C.Stefan_Boltzmann

# ----- fonctions utiles -----
def planck_lambda_per_um(wl_um, T):
    """
    Radiance spectrale de Planck par µm.
    
    wl_um : scalaire ou array (µm)
    T     : scalaire ou array (K)
    
    Retourne W·m^-2·sr^-1·µm^-1
    """
    wl_um = np.atleast_1d(wl_um)
    T = np.atleast_1d(T)

    # Conversion en mètres
    wl_m = wl_um * 1e-6  # µm -> m

    # Calcul du facteur exponentiel
    x = (C.h * C.c) / (wl_m[:, None] * C.k * T[None, :])  # broadcasting

    # Formule de Planck
    B_m = (2 * C.h * C.c**2) / (wl_m[:, None]**5 * np.expm1(x))  # W·m^-2·sr^-1·m^-1
    B_um = B_m * 1e-6  # conversion en µm^-1

    # Si T est scalaire, on retourne un vecteur 1D
    if B_um.shape[1] == 1:
        return B_um[:, 0]
    return B_um


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

    # interpolation sur la grille demandée
    rad_interp = np.interp(wvl_um, wvl_file_um, rad_file, left=0, right=0)

    # correction géométrique simple (Cas Lambertien)
    mu_i = np.cos(np.radians(i))
    mu_e = np.cos(np.radians(e))
    return rad_interp * mu_i * mu_e

# ----- paramètres à fixer pour la simulation -----
periode = "jour"   # "jour" ou "nuit"
Tmax = 1600.0      # K (centre du hotspot)
Tmin = 130.0       # K (fond)
d_MAJIS = 57e3     # m (77 km/px ; mettre 57e3 pour jour)
r0_list_km = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 5.0]  # liste des r0 à tester en km
r0_list = [r * 1e3 for r in r0_list_km] #on met les r0 en mètres pour faciliter les calculs suivants
wl_um = np.linspace(0.7016, 5.018, 300) #domaine spectral voulu (µm)

# stockage des résultats
L_pixel_list = []
T_eff_list = []

# Calcul de la radiance de fond (background) (interpolation des données NIMS et JIRAM)
if periode == "jour":
        wvl_all_um = wl_um
        io_rad_nims = get_radiance(wvl_all_um, i=14.13, e=0, filename="Io_NIMS.csv")
        io_rad_jiram = get_radiance(wvl_all_um, i=14.13, e=0, filename="Io_JIRAM_JM0261.csv")

        # normalisation entre spectres (méthode Clément)
        mask = (wvl_all_um > 2.5) & (wvl_all_um < 5.0)
        norm_factor = np.mean(io_rad_jiram[mask]) / np.mean(io_rad_nims[mask])
        L_background = io_rad_nims * norm_factor

else:
        L_background = planck_lambda_per_um(wl_um, Tmin)

# ----- boucle sur r0 pour calcul de chaque T_eff -----
#méthode : 
# 1ère étape : Pour chaque r0, on calcule le spectre de T(r) de r = 0 jusqu'à r = d_MAJIS (= taille du pixel) (loi de Planck)
# avec T(r) = (Tmax - Tmin) * exp(- (r^2) / (r0^2)) + Tmin
# 2ème étape : ensuite on rajoute la radiance de fond (background)
# 3ème étape : on calcule T_eff du pixel avec la radiance totale (hotspot + background) (loi de Stefan-Boltzmann)

for r0 in r0_list:
    # grille radiale de 0 à r0
    nr = 10000 # nombre de points radiaux
    r = np.linspace(0.0, d_MAJIS, nr) #on intègre le profil B_lambda(T(r)) sur un rayon égal à la taille du pixel
    dr = r[1] - r[0]
    dA_ring = 2.0 * np.pi * r * dr
    area_pixel = 2 * np.pi * d_MAJIS**2 #on normalise par l'aire du pixel (disque de rayon d_MAJIS)

    # profil T(r)
    T_r = (Tmax - Tmin) * np.exp(- (r**2) / (r0**2)) + Tmin

    # radiance hotspot moyenne (intégration radiale de B_lambda(T(r)))
    B_wl_r = planck_lambda_per_um(wl_um, T_r)   # shape (n_wl, nr)
    numer = np.sum(B_wl_r * dA_ring, axis=1)  # (n_wl,)
    L_hotspot_avg = numer / area_pixel   # radiance moyenne sur le hotspot

    # somme reflectance + hotspot
    L_pixel = L_background + L_hotspot_avg

    # T_eff
    integrated_radiance_sr = np.trapezoid(L_pixel, wl_um)
    exitance = np.pi * integrated_radiance_sr
    T_eff = (exitance / sigma) ** 0.25

    # stockage
    L_pixel_list.append(L_pixel)
    T_eff_list.append(float(T_eff))

# ----- affichage -----
print("r0 (km)      T_eff (K)")
for r0_km, Te in zip(r0_list_km, T_eff_list):
    print(f"{r0_km:6.2f}     {Te:8.2f}")

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
    r_km = np.logspace(-3, np.log10(d_MAJIS/1000), 300)  # de 1e-3 km (=1 m) à 5*r0
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

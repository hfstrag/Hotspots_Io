import Fonctions as F
import main as M
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C
import pandas as pd

# --- Lecture des données Veeder et al. (2015) ---
veeder_data = pd.read_csv("Veeder.txt", sep=r"\s+", engine="python")

#Conversion en numérique et nettoyage
veeder_data["Teff(K)"] = pd.to_numeric(veeder_data["Teff(K)"], errors="coerce")
veeder_data["Power(GW)"] = pd.to_numeric(veeder_data["Power(GW)"], errors="coerce")

# Grille de recherche pour r0
r0_grid_km = np.logspace(-2, 1.5, 100)
r0_estimates = []

#Boucle sur les observations
estimated_ro_list = []
for i in range(len(veeder_data)):
    row = veeder_data.iloc[i]
    Teff_obs = row["Teff(K)"]
    P_obs_W = row["Power(GW)"] * 1e9  # conversion GW -> W
    best, diag = F.estimate_r0_from_obs(Teff_obs, P_obs_W, 1600, r0_grid_km, verbose=False)
    estimated_ro_list.append(best["r0_km"])

# Ajout au DataFrame
veeder_data["Estimated_r0_km"] = estimated_ro_list
print("\nTable augmentée du rayon du volcan:")
print(veeder_data[["Name", "Teff(K)", "Power(GW)", "Estimated_r0_km"]])

# --- Graphique Teff vs Power avec r0 estimé en couleur ---
plt.figure(figsize=(8,6))
sc = plt.scatter(
    veeder_data["Teff(K)"],
    veeder_data["Power(GW)"],
    c=veeder_data["Estimated_r0_km"],
    cmap="viridis",
    s=80,
    edgecolor="k"
)
plt.colorbar(sc, label="r0 estimé (km)")
plt.xlabel("Température effective Teff (K)")
plt.ylabel("Puissance (GW)")
plt.title("Comparaison données Veeder (2015) et r0 estimés")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.savefig("Teff_vs_Power.png", dpi=300)
plt.show()
plt.close()
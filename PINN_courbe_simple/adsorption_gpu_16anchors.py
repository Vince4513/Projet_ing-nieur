# -----------------------------------------------------------------------
# Importation des bibliothèques
# -----------------------------------------------------------------------
import numpy as np # manipulation des tableaux
import pandas as pd # manipulation des dataframes
import deepxde as dde # bibliothèques de deep learning basé sur tensorflow
from datetime import datetime # gestion du temps
import matplotlib.pyplot as plt # affichage des graphes
import imageio.v2 as imageio # création de vidéos à partir d'images
from io import BytesIO # lecture d'images dans le buffer

dde.config.set_random_seed(48) # Configuration de la graine
dde.config.set_default_float('float64') # Configuration du type de float

global x, t               # Accès aux variables dans tous le code (y compris les fonctions)

ug = 0.01                 # Vitesse spécifique de gaz
epsb = 0.5                # Porosité du lit
kg = 0.0001               # Coefficient de transfert
Ke = 10.0                 # Coefficient d'équilibre
dp = 0.005                # Diamètre de particule
as_ = 6*(1-epsb)/dp       # Surface spécifique, dépend du diamètre de la sphère

x_start = 0               # distance de départ en mètres
x_end = 1                 # distance de fin en mètres
time_start = 0            # temps de départ en secondes
time_end = 1              # temps de fin en secondes

total_points = 200        # Nb de points dans l'intervalle de distance (x_end - x_start)
num_time_steps = 100      # Nb de points dans l'intervalle de temps (time_end - time_start)

iterations = 15000        # Nombre d'itérations du modèle avec l'optimizer Adam

# Fonction lançant l'apprentissage et la prediction
# Résultat:
# cs: concentration prédite
# cg: concentration prédite
# loss_history, train_state: les résultats du modèle
# input, output: entrées et sorties du PINN
now = datetime.today()    # Stockage de la date (format: aaaa_mm_jj_hh_mm_ss)
datetime_ = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}" # création chaine de caractères indiquant la date

# Préparation des données pour la prédiction à l'aide de PINN -------------------------------
x = np.linspace(x_start, x_end, total_points)                 # Créer un vecteur x de longueur total_points découpant l'intervalle x_start à x_end
t = np.linspace(time_start, time_end, num_time_steps)         # Créer un vecteur t de longueur num_time_steps découpant l'intervalle time_start à time_end
X, T = np.meshgrid(x,t)                                       # Créer 2 matrices dupliquant les vecteurs x et t du nombre de valeurs de l'autre vecteur

# Necessite d'avoir les deux csv dans son repertoire où l'on execute le code
cs_analytical = pd.read_csv('cs_analytical.csv', header=None) # Chargement des données analytiques de cs
cg_analytical = pd.read_csv('cg_analytical.csv', header=None) # Chargement des données analytiques de cg

# Convert DataFrame to arrays
cs_analytical = cs_analytical.values # Converti le dataframe des valeurs analytiques de cs en tableau numpy
cg_analytical = cg_analytical.values # Converti le dataframe des valeurs analytiques de cg en tableau numpy

observe_x = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # Combine les 2 vecteurs au format (20000, 2)
observe_cs = np.hstack((cs_analytical.flatten()[:,None], T.flatten()[:,None])) # Mise au meme format que observe_x (20000, 2)
observe_cg = np.hstack((cg_analytical.flatten()[:,None], T.flatten()[:,None])) # Mise au meme format que observe_x (20000, 2)


# idx_CS = np.random.choice(observe_cs.shape[0], 16, replace=False) # Selection de 16 temps aléatoirement
# idx_CS[0] = 200 # imposer la 1ère mesure de cs en début de simulation x=0 tous les 200 valeurs

# pour trouver l'indice de distance correspondant : id_x = x*199
# pour trouver l'indice de cs avec le temps correspondant : id_cs = id_x + 200*100*t
idx_CS = np.array([0, 40, 80, 199, # t = 0
                   5000, 5040, 5080, 5199, # t = 0.2
                   7000, 7040, 7080, 7199, # t = 0.4
                   10000, 10040, 10080, 10199], # t = 1
                  dtype="int64") # imposer les 16 valeurs de temps

obs_cs = observe_cs[idx_CS, 0:] # Selection dans x des 16 échantillons
obs_xcs = observe_x[idx_CS, 0:] # Selection dans cs analytique des 16 échantillons

# idx_CG = np.random.choice(observe_cg.shape[0], 16, replace=False) # Selection de 16 temps aléatoirement

# pour trouver l'indice de distance correspondant : id_x = x*199
# pour trouver l'indice de cs avec le temps correspondant : id_cg = id_x + 200*100*t
idx_CG = np.array([200, 240, 280, 399, # t = 0.1
                   5000, 5040, 5080, 5199, # t = 0.2
                   7000, 7040, 7080, 7199, # t = 0.4
                   10000, 10040, 10080, 10199], # t = 1
                  dtype="int64") # imposer les 16 valeurs de temps

obs_cg = observe_cg[idx_CG, 0:] # Selection dans x des 16 échantillons
obs_xcg = observe_x[idx_CG, 0:] # Selection dans cg analytique des 16 échantillons

# Système d'équations différentielles partielles -------------------------------
def pde(z, u):
  #z is a 2D array where z[:,0:1] represents spatial domain in z direction and z[:,1:2] represents the temporal domain
  #z is the input of the neural net
  #u is a 2D array where u[:,0:1] represents water depth and u[:,1:2] represents the water velocity
  #u is the output of the neural net

  cs, cg = u[:, 0:1], u[:, 1:] # Récupération des valeurs de cs et cg

  dcs_dt = dde.grad.jacobian(u, z, i=0, j=1) # Calcul du gradient dcs/dt
  dcg_dz = dde.grad.jacobian(u, z, i=1, j=0) # Calcul du gradient dcg/dz
  dcg_dt = dde.grad.jacobian(u, z, i=1, j=1) # Calcul du gradient dcg/dt

  eq1 = (dcg_dt + ug * dcg_dz) + (kg*as_/epsb) * (cg-cs/Ke) # Première équation à respecter
  eq2 = dcs_dt - (kg*as_/(1-epsb)) * (cg-cs/Ke)             # Deuxième équation à respecter

  return [eq1, eq2] # Retour des résultats des deux équations



# Fonctions des conditions initiales de concentration Cs et Cg ------------------------------
def boundary_l(x,on_boundary):
  return on_boundary  and np.isclose(x[0], x_start)  # définit le boundary x=0, left

def boundary_r(x,on_boundary):
  return on_boundary  and np.isclose(x[0], x_end)  # définit le boundary x=L, right


geom = dde.geometry.Interval(x_start, x_end) # définition de l'espace de l'experience
timedomain = dde.geometry.TimeDomain(time_start, time_end) # définition du temps de l'experience
geomtime = dde.geometry.GeometryXTime(geom, timedomain) # définition de l'espace temps de l'experience


#Définir les conditions initiales et les conditions aux limites -----------------------
ic_cs = dde.icbc.IC(
  geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=0
) # Definition des conditions initiales de cs
ic_cg = dde.icbc.IC(
  geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=1
) # Definition des conditions initiales de cg
bc_cg = dde.icbc.DirichletBC(
  geomtime, lambda x: 1, boundary_l, component=1
) # Definition des conditions aux frontières de cg: C_g(0,t)=1
bc_dcg = dde.icbc.OperatorBC(
  geomtime, lambda x, y, _ : dde.grad.jacobian(y, x, i=0, j=1), boundary_r
) # Definition des conditions aux frontières de cg: dC_g/dz(L,t)=0

observe_cs = dde.icbc.PointSetBC(obs_xcs, obs_cs, component = 0)
observe_cg = dde.icbc.PointSetBC(obs_xcg, obs_cg, component = 1)

data = dde.data.TimePDE(
  geomtime, # Fixe l'espace temps de l'experience
  pde, # Fixe le système d'équations différentielles
  [ic_cs, ic_cg, bc_cg, bc_dcg, observe_cs, observe_cg], # Fixe toutes les IC et BC
  num_domain=350000, # définition du nb de points dans l'espace total
  num_boundary=10000, # définition du nb de points aux frontières
  num_initial=30000 # définition du nb de points à t = 0
)

layer_size = [2] + [64] * 4 + [2] # Créer le réseau en précisant le nb de couches et neurones
net = dde.nn.FNN(layer_size, "tanh", "Glorot normal") # Détermine l'initialisation des poids du réseau et la fonction d'activation utilisée
model = dde.Model(data, net) # Combine le réseau défini avec les toutes les conditions et points définis



# Adam --------------------------------------------------------------------------
model.compile("adam", lr=1e-4, loss_weights = [1e3, 1e8, 1e0, 1e0, 1e3, 1e0, 1e3, 1e3]) # Défini l'optimiser, le learning rate et les poids accordés à chaque équations/IC/BC/PointSetBC
losshistory, train_state = model.train(iterations= iterations) # Entrainement du réseau
dde.saveplot(losshistory, train_state, issave=True) # Affiche les graphes d'entrainement du reséau
model.save(f"Run_{datetime_}_it{iterations}/1stStageAdamOptimized.h5") # Sauvegarde le modèle entrainé
# model.restore("Run_2024_1_18_11_52_5_it15000/1stStageAdamOptimized.h5-15000.ckpt")

### Post Processing ------------------------------------------------------------
output = model.predict(observe_x) # A partir de données d'entrée, on obtient les résultats provenant du modèle après Adam

# Extracting concentration 'cs' and concentration 'cg' from the result ------------------
cs = (output[:, 0].reshape((-1, total_points)).T) # Extraction de la concentration cs que l'on reformate en vecteur
cg = (output[:, 1].reshape((-1, total_points)).T) # Extraction de la concentration cg que l'on reformate en vecteur

np.save(f"Run_{datetime_}_it{iterations}/1stStageAdamOptimized_cs.npy", cs) # Sauvegarde de cs après entrainement avec Adam
np.save(f"Run_{datetime_}_it{iterations}/1stStageAdamOptimized_cg.npy", cg) # Sauvegarde de cg après entrainement avec Adam



# # LBFGS --------------------------------------------------------------------------
# dde.optimizers.set_LBFGS_options(maxiter= 10000, gtol= 1e-5) # Défini le nb max d'itérations et le learning rate
# model.compile("L-BFGS", loss_weights = [1e3, 1e8, 1e0, 1e0, 1e3, 1e0, 1e3, 1e3]) # Défini l'optimiser et les poids accordés à chaque équations/IC/BC
# losshistory, train_state = model.train() # Entrainement du réseau
# dde.saveplot(losshistory, train_state, issave=True) # Affiche les graphes d'entrainement du reséau
# model.save(f"Run_{datetime_}_it{iterations}/2ndStage_LBFGS_Optimzed.h5") # Sauvegarde le modèle entrainé
# # model.restore("Run_2024_1_18_11_52_5_it15000/2ndStage_LBFGS_Optimzed.h5-15000.ckpt")

# ### Post Processing ------------------------------------------------------------
# output = model.predict(observe_x) # A partir de données d'entrée, on obtient les résultats provenant du modèle après LBFGS

# # Extracting concentration 'cs' and concentration 'cg' from the result ------------------
# cs = (output[:, 0].reshape((-1, total_points)).T) # Extraction de la concentration cs que l'on reformate en vecteur
# cg = (output[:, 1].reshape((-1, total_points)).T) # Extraction de la concentration cg que l'on reformate en vecteur

# np.save(f"Run_{datetime_}_it{iterations}/2ndStage_LBFGS_Optimzed_cs.npy", cs) # Sauvegarde de cs après entrainement avec Adam puis LBFGS
# np.save(f"Run_{datetime_}_it{iterations}/2ndStage_LBFGS_Optimzed_cg.npy", cg) # Sauvegarde de cg après entrainement avec Adam puis LBFGS



# Graphs -----------------------------------------------------------------------
# Functions
def mean_squared_error(predicted_values, true_values):
    """
    Calculate the mean squared error between predicted and true values.

    Parameters:
    - predicted_values: numpy array, predicted concentrations from the neural network.
    - true_values: numpy array, true concentrations from analytical solutions.

    Returns:
    - mse: float, mean squared error.
    """
    mse = np.mean((predicted_values - true_values)**2)
    rmse = np.sqrt(mse)
    return rmse



rmse_cs = mean_squared_error(cs, cs_analytical) # Stocke la RMSE entre l'analytique et le prédit de la concentration cs
rmse_cg = mean_squared_error(cg, cg_analytical) # Stocke la RMSE entre l'analytique et le prédit de la concentration cg


dt = time_end / num_time_steps # Détermine l'intervalle de décalage de temps entre chaque image

### Graphes de la concentration Cs -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 3.5)) # Défini les axes et la taille de l'image

frames = [] # Créer une liste pour stocker les images générées

for timestep in range(cs.shape[1]): # Boucle sur chaque pas de temps
    ax.clear() # Effacer l'axe

    ax.plot(x, cs_analytical[:, timestep], label='Analytical') # Tracer la concentration de cs analytique au pas de temps actuel
    ax.plot(x, cs[:, timestep], linestyle='--', label='PINN') # Tracer la concentration de cs prédite au pas de temps actuel

    ax.fill_between(x, 0, cs[:, timestep], color='skyblue', alpha=0.5) # Remplir la zone entre les courbes

    timestamp = (timestep+1) * dt # Décalage du temps d'une valeur dt
    ax.set_xlabel('x-distance [m]') # Définir l'étiquette de l'axe x
    ax.set_ylabel('Concentration cs [g/mol]') # Définir l'étiquette de l'axe y
    ax.set_title(f'Time: {timestamp:.2f} s') # Définir le titre du graphe
    ax.set_xlim([x_start, x_end]) # Définir les dimensions de l'axe x
    ax.set_ylim([0, 10.2]) # Définir les dimensions de l'axe y
    ax.legend()  # Ajout de la légende

    img_buffer = BytesIO() # Créer un objet fichier en mémoire

    plt.savefig(img_buffer, format='png') # Sauvegarde de la figure actuelle dans le fichier en mémoire objet

    img_buffer.seek(0) # Lire le contenu de l'objet fichier en mémoire
    img_data = img_buffer.getvalue() # Lire le contenu de l'objet fichier en mémoire
    img = imageio.imread(img_data, format='PNG') # Lire le contenu de l'objet fichier en mémoire
    frames.append(img) # Ajout de l'image à la liste frames

    img_buffer.close() # Effacer l'objet fichier en mémoire pour la prochaine itération

mp4_filename = f'Run_{datetime_}_it{iterations}/cs_animation.mp4' # Donner un nom au fichier MP4 avec l'emplacement
imageio.mimsave(mp4_filename, frames, fps=10) # Enregistrer la liste des images dans un fichier MP4

plt.show() # Montrer l'animation finale de cs


### Graphes de la concentration Cg ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.5, 3.5)) # Défini les axes et la taille de l'image

frames = [] # Créer une liste pour stocker les images générées

for timestep in range(cg.shape[1]): # Boucle sur chaque pas de temps
    ax.clear() # Effacer l'axe

    ax.plot(x, cg_analytical[:, timestep], label='Analytical') # Tracer la concentration de cg analytique au pas de temps actuel
    ax.plot(x, cg[:, timestep], linestyle='--', label='PINN') # Tracer la concentration de cg prédite au pas de temps actuel

    ax.fill_between(x, 0, cg[:, timestep], color='skyblue', alpha=0.5) # Remplir la zone entre les courbes

    timestamp = (timestep+1) * dt# Décalage du temps d'une valeur dt
    ax.set_xlabel('x-distance [m]') # Définir l'étiquette de l'axe x
    ax.set_ylabel('Concentration cg [g/mol]') # Définir l'étiquette de l'axe y
    ax.set_title(f'Time: {timestamp:.2f} s') # Définir le titre du graphe
    ax.set_xlim([x_start, x_end]) # Définir les dimensions de l'axe x
    ax.set_ylim([0, 1.2]) # Définir les dimensions de l'axe y
    ax.legend()  # Ajout de la légende

    img_buffer = BytesIO() # Créer un objet fichier en mémoire

    plt.savefig(img_buffer, format='png') # Sauvegarde de la figure actuelle dans le fichier en mémoire objet

    img_buffer.seek(0) # Lire le contenu de l'objet fichier en mémoire
    img_data = img_buffer.getvalue() # Lire le contenu de l'objet fichier en mémoire
    img = imageio.imread(img_data, format='PNG') # Lire le contenu de l'objet fichier en mémoire
    frames.append(img) # Ajout de l'image à la liste frames

    img_buffer.close() # Effacer l'objet fichier en mémoire pour la prochaine itération

mp4_filename = f'Run_{datetime_}_it{iterations}/cg_animation.mp4'# Donner un nom au fichier MP4 avec l'emplacement
imageio.mimsave(mp4_filename, frames, fps=10) # Enregistrer la liste des images dans un fichier MP4

plt.show() # Montrer l'animation finale de cg
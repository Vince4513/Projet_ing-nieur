# pinn.py

# -----------------------------------------------------------------------
# Importation des bibliothèques
# -----------------------------------------------------------------------
import gspread
import numpy as np
import pandas as pd
import deepxde as dde
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PINN:
    
    # Configuration de la graine et du type de float
    dde.config.set_random_seed(48)
    dde.config.set_default_float('float64')
    
    # Fonction d'initialisation du PINN modélisant un réacteur de longueur x_end - x_start avec une temporalité de time_end - time_start
    # On définit également le nombre de points dans ces deux intervalles
    def __init__(self, x_start, x_end, time_start, time_end, total_points, num_time_steps, save_path):
        self.ug = 0.01                      #Vitesse spécifique de gaz
        self.epsb = 0.5                     #Porosité du lit
        self.kg = 0.0001                    #Coefficient de transfert
        self.Ke = 10.0                      #Coefficient d'équilibre
        self.dp = 0.005                     #Diamètre de particule
        self.as_ = 6*(1-self.epsb)/self.dp  #Surface spécifique, dépend du diamètre de la sphère
        
        self.L_scale = x_end - x_start         # longueur du lit
        self.T_scale = time_end - time_start   # durée expérience
        self.U_scale = 3 #put tentative maximum value you think is appropriate in m/s

        #Scaling the variables 
        self.x_start = x_start/ self.L_scale
        self.x_end = x_end / self.L_scale
        self.time_start = time_start / self.T_scale
        self.time_end = time_end / self.T_scale
        self.total_points = total_points
        self.num_time_steps = num_time_steps
        self.save_path = save_path

    # Fonction lançant l'apprentissage et la prediction
    # Résultat: 
    # cs: concentration prédite 
    # cg: concentration prédite
    # loss_history, train_state: les résultats du modèle
    # input, output: entrées et sorties du PINN
    def run(self, current_params, loss_weights, loss_weights_lbfgs, iterations):

        now = datetime.today()
        datetime_ = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        
        # Preparation of Input for Prediction using PINN -------------------------------
        x = np.linspace(self.x_start, self.x_end, self.total_points)
        t = np.linspace(self.time_start, self.time_end, self.num_time_steps)
        X, T = np.meshgrid(x,t)
        input = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

        # Fonction créant les équations différentielles partielles --------------------------
        def pde(z, u):
            #z is a 2D array where z[:,0:1] represents spatial domain in z direction and z[:,1:2] represents the temporal domain
            #z is the input of the neural net
            #u is a 2D array where u[:,0:1] represents water depth and u[:,1:2] represents the water velocity
            #u is the output of the neural net

            cs, cg = u[:, 0:1], u[:, 1:]

            dcs_dt = dde.grad.jacobian(u, z, i=0, j=1)
            dcg_dz = dde.grad.jacobian(u, z, i=1, j=0)
            dcg_dt = dde.grad.jacobian(u, z, i=1, j=1)

            # eq1 = (dcg_dt + ug * dcg_dz) + (kg*as_/epsb) * (cg-cs/Ke)
            # eq2 = dcs_dt - (kg*as_/(1-epsb)) * (cg-cs/Ke)
            eq1 = (((self.L_scale * dcg_dt) / (self.T_scale * self.U_scale)) + self.ug * dcg_dz) + (self.kg*self.as_/self.epsb) * (10*cg - cs)/10
            eq2 = ((self.L_scale * dcs_dt) / (self.T_scale * self.U_scale)) - (self.kg*self.as_/(1-self.epsb)) * (10*cg - cs)/10

            return [eq1, eq2]

        #  function for  Initial condition of water depth ------------------------------
        def boundary_l(x,on_boundary):
            return on_boundary  and np.isclose(x[0], self.x_start)  # définit le boundary x=0, left

        def boundary_r(x,on_boundary):
            return on_boundary  and np.isclose(x[0], self.x_end)  # définit le boundary x=L, right


        geom = dde.geometry.Interval(self.x_start, self.x_end)
        timedomain = dde.geometry.TimeDomain(self.time_start, self.time_end)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)

        #Define initial condition and boundary condition --------------------------
        ic_cs = dde.icbc.IC(
            geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=0
        )
        ic_cg = dde.icbc.IC(
            geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=1
        )
        bc_cg = dde.icbc.DirichletBC(
            geomtime, lambda x: 1, boundary_l, component=1
        ) # C_g(0,t)=1
        bc_dcg = dde.icbc.OperatorBC(
            geomtime, lambda x, y, _ : dde.grad.jacobian(y, x, i=0, j=1), boundary_r
        ) # dc_g/dz(L,t)=0

        # Anchors of both concentrations ---------------------------------------
        # observe_cs = dde.icbc.PointSetBC(
        #     np.array([[0.0, 0.3, 0.5, 0.8, 0.0, 0.3, 0.5, 0.8, 0.0, 0.3, 0.5, 0.8, 0.0, 0.3, 0.5, 0.8],
        #               [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0]]).T,
        #     np.array([[10.0, 8.0, 6.0, 5.0, 9.0, 7.5, 6.0, 5.0, 9.0, 7.0, 6.0, 4.0, 9.0, 7.0, 5.0, 3.0],
        #               [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0]]).T, component=0)
        # observe_cg = dde.icbc.PointSetBC(
        #     np.array([[0.0, 0.3, 0.5, 0.8, 0.0, 0.3, 0.5, 0.8, 0.0, 0.3, 0.5, 0.8, 0.0, 0.3, 0.5, 0.8],
        #               [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0]]).T,
        #     np.array([[1.0, 0.6, 0.5, 0.4, 1.0, 0.6, 0.5, 0.4, 0.8, 0.6, 0.5, 0.4, 0.8, 0.5, 0.4, 0.3],
        #               [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0]]).T, component=1)

        data = dde.data.TimePDE(
            geomtime, pde, [ic_cs, ic_cg, bc_cg, bc_dcg], 
            num_domain=current_params["num_domain"], 
            num_boundary=current_params["num_boundary"], 
            num_initial=current_params["num_initial"], 
            anchors=input
        )
        
        layer_size = [2] + [current_params["nb_neur_couche"]] * current_params["nb_couches"] + [2]
        net = dde.nn.FNN(layer_size, current_params["activation"], current_params["initializer"])
        model = dde.Model(data, net)

        # Adam --------------------------------------------------------------------------
        model.compile("adam", lr=3e-4, loss_weights = loss_weights)
        pde_resampler = dde.callbacks.PDEPointResampler(period=10)
        losshistory, train_state = model.train(iterations= iterations, callbacks=[pde_resampler])
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        # Save the trained model
        model.save(f"{self.save_path}MODELS/{datetime_}_1stStageAdamOptimized.h5")

        # LBFGS --------------------------------------------------------------------------
        dde.optimizers.set_LBFGS_options(maxiter= 100, gtol= 1e-5)
        model.compile("L-BFGS", loss_weights = loss_weights_lbfgs)    
        losshistory, train_state = model.train(callbacks=[pde_resampler])
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        # Save the trained model
        model.save(f"{self.save_path}MODELS/{datetime_}_2ndStage_LBFGS_Optimzed.h5")

        ### Post Processing ------------------------------------------------------------
        output = model.predict(input) #passing the input gridpoints and timestamps through optimized neural network
    
        # Extracting water depth 'cs' and velocity 'cg' from the result ------------------
        cs = (output[:, 0].reshape((-1, self.total_points)).T)
        cg = (output[:, 1].reshape((-1, self.total_points)).T) * self.U_scale

        np.save(f"{self.save_path}MODELS/{datetime_}_cs.npy", cs)
        np.save(f"{self.save_path}MODELS/{datetime_}_cg.npy", cg)
        
        return losshistory, train_state

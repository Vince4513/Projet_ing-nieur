import numpy as np

from pinn import PINN

params_to_search = {
    # ["Hammersley", "LHS", "Halton", "pseudo", "Sobol", "uniform"]
    "train_distribution": ["uniform"],
    "num_domain": [5000],
    "num_boundary":[5000],
    "num_initial": [1000],
    "num_test": [1000],

    # ["Glorot normal", "Glorot uniform", "He normal", "He uniform", "LeCun normal", "LeCun uniform", "Orthogonal", "zeros"]
    "initializer": ["Glorot normal"],
    # [elu, relu, gelu, selu, sigmoid, silu, sin, silu, tanh]
    "activation": ["tanh"],
    "nb_couches": [4],
    "nb_neur_couche": [60]

    # "maxcor": [100],
    # "ftol": [0],
    # "gtol": [1e-8],
    # "maxiter": [15000],
    # "maxfun": None,
    # "maxls":  [100]
}

# Choix des paramètres parmi les valeurs porposées
current_params = {param: np.random.choice(values) if isinstance(values, list) else values for param, values in params_to_search.items()}

loss_weights = [1e2, 1e2, 1e5, 1e5, 1e4, 1e4]
loss_weights_lbfgs = [1e2, 1e2, 1e5, 1e5, 1e4, 1e4]



# Create an instance of PINN 
pinn_obj = PINN(
    x_start = 0, 
    x_end = 1, 
    time_start = 0, 
    time_end = 1, 
    total_points = 200,
    num_time_steps = 100,
    save_path=""
)


# Run the PINN
losshistory, train_state = pinn_obj.run(
    current_params = current_params, 
    loss_weights = loss_weights,
    loss_weights_lbfgs = loss_weights_lbfgs,
    iterations = 10000
)




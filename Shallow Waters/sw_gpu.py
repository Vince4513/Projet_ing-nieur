import numpy as np # manipulation des tableaux
import pandas as pd # manipulation des dataframes
import deepxde as dde # bibliothèques de deep learning basé sur tensorflow
from datetime import datetime # gestion du temps
import matplotlib.pyplot as plt # affichage des graphes
import imageio.v2 as imageio # création de vidéos à partir d'images
from io import BytesIO # lecture d'images dans le buffer

dde.config.set_random_seed(48) # Configuration de la graine
dde.config.set_default_float('float64') # Configuration du type de float 

#Configuration for the simulation
x_start = 0 #Starting point of domain
x_end = 100 #End point of domain
time_start = 0 #Starting time of simulation
time_end = 8 #Ending time of simulation
g = 9.81  # Specify the value for gravity constant g
dam_location = 35 #Location of dam
dam_height = 1  #water depth upstream of dam
downstream_depth = 0.4  #water depth downstream of dam

#Definition of Scaling factors
L_scale = x_end - x_start
T_scale = time_end - time_start
U_scale = 3 #put tentative maximum value you think is appropriate in m/s
H_scale = dam_height
G_scale = (U_scale**2) / H_scale

#Scaling the variables
x_start = x_start/L_scale
x_end = x_end / L_scale
time_start = time_start / T_scale
time_end = time_end / T_scale
g = g / G_scale  # Specify the value for gravity constant g
dam_location = dam_location / L_scale
dam_height = dam_height / H_scale  #water depth upstream of dam
downstream_depth = downstream_depth / H_scale  #water depth downstream of dam

iterations = 20000 #iterations for optimal neural net

now = datetime.today()    # Stockage de la date (format: aaaa_mm_jj_hh_mm_ss)
datetime_ = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}" # création chaine de caractères indiquant la date

# Preparation of Input for Prediction using PINN
total_points = 200 #Number of gridpoints in domain
num_time_steps = 100 #Number of timesteps
x = np.linspace(x_start, x_end, total_points)
t = np.linspace(time_start, time_end, num_time_steps)
X, T = np.meshgrid(x,t)
input = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

#Define the simplified 1D Shallow Water Equations
def pde(x, y):
    #x is a 2D array where x[:,0:1] represents spatial domain in x direction and x[:,1:2] represents the temporal domain
    #x is the input of the neural net
    #y is a 2D array where y[:,0:1] represents water depth and y[:,1:2] represents the water velocity
    #y is the output of the neural net

    h, u = y[:, 0:1], y[:, 1:]

    dh_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dh_dt = dde.grad.jacobian(y, x, i=0, j=1)
    du_dx = dde.grad.jacobian(y, x, i=1, j=0)
    du_dt = dde.grad.jacobian(y, x, i=1, j=1)

    mass_balance = ((L_scale * dh_dt) / (T_scale * U_scale))  + ( h * du_dx + u * dh_dx )
    momentum_x = ((L_scale * du_dt) / (T_scale * U_scale)) + u * du_dx + g * dh_dx #+ g * S0 #+ (g * n_st ** 2 * u**2 * h**(-4/3))

    return [mass_balance, momentum_x]


# function for  Initial condition of water depth
def h_initial(x):
    h = np.where(x < dam_location, dam_height, downstream_depth)
    return h


geom = dde.geometry.Interval(x_start, x_end)
timedomain = dde.geometry.TimeDomain(time_start, time_end)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#Define initial condition of water depth and velocity
ic_h = dde.icbc.IC(
    geomtime, lambda x: h_initial(x[:, 0:1]), lambda _, on_initial: on_initial, component=0
)
ic_u = dde.icbc.IC(
    geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=1
)

data = dde.data.TimePDE(
    geomtime, pde, 
    [ic_h, ic_u], 
    num_domain=350000, 
    num_initial=30000
)

net = dde.nn.FNN([2] + [50] * 8 + [2], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Adam --------------------------------------------------------------------------
#Compile the neural net with Adam optimizer, set learning rate and loss weight to mass balance equation, momentum equation, initial condition of h and initial condition of u respectively
model.compile("adam", lr=5e-4, loss_weights = [1e2, 1e2, 1e5, 1e4])
pde_resampler = dde.callbacks.PDEPointResampler(period=10)
losshistory, train_state = model.train(iterations=iterations, callbacks=[pde_resampler])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# Save the trained model
model.save(f"SW_Run_{datetime_}_it{iterations}/1stStageAdamOptimized.h5") # Sauvegarde le modèle entrainé

### Post Processing
output = model.predict(input) #passing the input gridpoints and timestamps through optimized neural network

# Extracting water depth 'h' and velocity 'u' from the result
h = (output[:, 0].reshape((-1, total_points)).T) * H_scale
u = (output[:, 1].reshape((-1, total_points)).T) * U_scale

np.save(f"SW_Run_{datetime_}_it{iterations}/1stStageAdamOptimized_h.npy", h)
np.save(f"SW_Run_{datetime_}_it{iterations}/1stStageAdamOptimized_u.npy", u)

# LBFGS --------------------------------------------------------------------------
dde.optimizers.set_LBFGS_options(maxiter= 100000, gtol= 1e-5)
model.compile("L-BFGS", loss_weights = [100, 100, 1e5, 1e4])
losshistory, train_state = model.train(callbacks=[pde_resampler])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# Save the trained model
model.save(f"SW_Run_{datetime_}_it{iterations}/2ndStage_LBFGS_Optimzed.h5") # Sauvegarde le modèle entrainé

### Post Processing
output = model.predict(input) #passing the input gridpoints and timestamps through optimized neural network

# Extracting water depth 'h' and velocity 'u' from the result
h = (output[:, 0].reshape((-1, total_points)).T) * H_scale
u = (output[:, 1].reshape((-1, total_points)).T) * U_scale

np.save(f"SW_Run_{datetime_}_it{iterations}/2ndStage_LBFGS_Optimzed_h.npy", h)
np.save(f"SW_Run_{datetime_}_it{iterations}/2ndStage_LBFGS_Optimzed_u.npy", u)

# Necessite d'avoir les deux csv dans son repertoire où l'on execute le code 
h_analytical = pd.read_csv('h_analytical.csv', header=None)
u_analytical = pd.read_csv('u_analytical.csv', header=None)

# Convert DataFrame to arrays
h_analytical = h_analytical.values
u_analytical = u_analytical.values

dt = time_end / num_time_steps

### Post Processing
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Create a list to store the generated frames
frames = []

# Loop through each timestep
for timestep in range(h.shape[1]):
    # Clear the axis
    ax.clear()

    # Plot the water depth at the current timestep

    ax.plot(x, h_analytical[:, timestep], label='Analytical')
    ax.plot(x, h[:, timestep], linestyle='--', label='PINN')

    # Fill the area between the curves
    ax.fill_between(x, 0, h[:, timestep], color='skyblue', alpha=0.5)
   # ax.fill_between(x, 0, h_values_transpose[:, timestep], color='lightgreen', alpha=0.5)

    timestamp = (timestep+1) * dt
    # Set the axis labels and title
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('Water depth h [m]')
    ax.set_title(f'Time: {timestamp:.2f} s')
    ax.set_xlim([x_start, x_end])
    ax.set_ylim([0, 1.2])
    ax.legend()  # Add legend

    # Create an in-memory file object
    img_buffer = BytesIO()

    # Save the current figure to the in-memory file object
    plt.savefig(img_buffer, format='png')

    # Read the contents of the in-memory file object and add it to the list of frames
    img_buffer.seek(0)
    img_data = img_buffer.getvalue()
    img = imageio.imread(img_data, format='PNG')
    frames.append(img)

    # Clear the in-memory file object for the next iteration
    img_buffer.close()

# Save the list of frames as an MP4 file
# (adjust the file name and parameters as needed)
mp4_filename = f'SW_Run_{datetime_}_it{iterations}/water_depth_animation.mp4'
imageio.mimsave(mp4_filename, frames, fps=10)

# Show the final animation
plt.show()

#Plot for velocity ---------------------------

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Create a list to store the generated frames
frames = []

# Loop through each timestep
for timestep in range(u.shape[1]):
    # Clear the axis
    ax.clear()
    
    # Plot the water depth at the current timestep
    
    ax.plot(x, u_analytical[:, timestep], label='Analytical')
    ax.plot(x, u[:, timestep], linestyle='--', label='PINN')
    
    timestamp = (timestep+1) * dt
    # Set the axis labels and title
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('Water velocity u [m/s]')
    ax.set_title(f'Time: {timestamp:.2f} s')
    ax.set_xlim([x_start, x_end])
    ax.set_ylim([0, 1.5])
    ax.legend()  # Add legend
    
    # Create an in-memory file object
    img_buffer = BytesIO()
    
    # Save the current figure to the in-memory file object
    plt.savefig(img_buffer, format='png')
    
    # Read the contents of the in-memory file object and add it to the list of frames
    img_buffer.seek(0)
    img_data = img_buffer.getvalue()
    img = imageio.imread(img_data, format='PNG')
    frames.append(img)
    
    # Clear the in-memory file object for the next iteration
    img_buffer.close()
    
# Save the list of frames as an MP4 file
# (adjust the file name and parameters as needed)
mp4_filename = f'SW_Run_{datetime_}_it{iterations}/water_velocity_animation.mp4'
imageio.mimsave(mp4_filename, frames, fps=10)

# Show the final animation
plt.show()

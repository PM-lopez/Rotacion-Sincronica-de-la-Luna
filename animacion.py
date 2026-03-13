# Se implementa el codigo anterior explicado en detalle en solucion.ipynb
########################################################################from metodos_para_edos import *
import numpy as np
import matplotlib.pyplot as plt
from metodos_para_edos import rk4_method_second_order_2D, rk4_method_second_order

G=6.67430e-11  
M=5.972e24     
a = 3.844e8  

def f(t, r, v):
    r_mag = np.sqrt(r[0]**2 + r[1]**2)
    return -G*M / r_mag**3 * np.array([r[0], r[1]])

r0 = (a,0)

# Tiempo de simulacion
t0 = 0
tf = 365* 5 * 24 * 3600 * 3


h = 24 * 3600 
v0 = (0, 1028)  

t_values, r_values, v_values = rk4_method_second_order_2D(f, t0, r0, v0, tf, h)


Rluna = 1.737e6
m = 7.342e22  
I1 = 0.95 * 2/5 * m * Rluna**2  
I2 = I3 = 2/5 * m * Rluna**2  
k = 7.0e27 

def f_rotacional(t,theta,w):
    x_actual = np.interp(t, t_values, r_values[:, 0]) 
    y_actual = np.interp(t, t_values, r_values[:, 1])
    r_mag = np.sqrt(x_actual**2 + y_actual**2)
    theta_r= np.arctan2(y_actual, x_actual)
    return -((3*G*M*(I2-I1))/(I3*r_mag**3)) * np. sin(2*(theta - theta_r))- (k*w)/I3

theta0 = 0
w0 = 1.5e-5 

_, theta_values, w_values = rk4_method_second_order(f_rotacional, t0, theta0, w0, tf, h)

############################################################################################


#########################
####### Animacion #######
#########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Transformamos a arrays los datos
t_values = np.array(t_values)
r_values = np.array(r_values)
theta_values = np.array(theta_values)

# Creamos el grafico
fig, ax = plt.subplots(figsize=(8,8))

# Tierra
ax.scatter(0, 0, color="blue", s=150, zorder=3, label="Tierra")
# Órbita de la Luna
ax.plot(r_values[:,0], r_values[:,1], color="gray", linestyle="--", alpha=0.3, label="Órbita")

# Figuras a animar:

# La Luna
luna_dot, = ax.plot([], [], color="gray", marker="o", markersize=15, zorder=4, label="Luna")

# La línea Tierra-Luna
linea_distancia, = ax.plot([], [], color="green", linestyle=":", linewidth=1, alpha=0.7)

#La flecha de orientación
flecha_orientacion = ax.quiver(0, 0, 0, 0, color="red", scale=1, scale_units="xy", angles="xy", width=0.010, zorder=5, label="Eje Principal")

# Configuración de ejes
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Dinámica Orbital y Rotacional de la Luna")
ax.legend(loc="upper right")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

longitud_flecha = a * 0.3

# Funciones de animaciones
def init():
    luna_dot.set_data([], [])
    linea_distancia.set_data([], [])
    flecha_orientacion.set_UVC(0, 0) 
    return luna_dot, linea_distancia, flecha_orientacion

def update(frame):
    x_luna = r_values[frame, 0]
    y_luna = r_values[frame, 1]
    
    theta = theta_values[frame]

    luna_dot.set_data([x_luna], [y_luna])
    linea_distancia.set_data([0, x_luna], [0, y_luna])

    flecha_orientacion.set_offsets(np.array([[x_luna, y_luna]]))
    
    u = longitud_flecha * np.cos(theta) 
    v = longitud_flecha * np.sin(theta)
    
    flecha_orientacion.set_UVC(u, v)
    
    return luna_dot, linea_distancia, flecha_orientacion

step = 1 
indices = range(0, len(t_values), step)
ani = FuncAnimation(fig, update, frames=indices, init_func=init, blit=True, interval=50)

plt.show()




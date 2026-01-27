import numpy as np

def rk4_method_second_order_2D(f, t0: float, r0: tuple, v0: tuple, tf: float, h: float = 0.0001) -> tuple:
    """Método de Runge-Kutta de cuarto orden (RK4) para resolver ecuaciones diferenciales de segundo orden en 2D.

    Este método usa el esquema de Runge-Kutta de cuarto orden, que es uno de los métodos más precisos
    para integrar ecuaciones diferenciales ordinarias. El método se aplica a sistemas de segundo orden
    en un espacio bidimensional (x, y), utilizando el campo de fuerzas o aceleraciones definidas por la función `f`.

    Args:
        f (callable): Función que define la aceleración (o fuerza/m), dependiente del tiempo,
                      la posición y la velocidad. Debe tener la forma `f(t, r, v) -> np.array([Fx, Fy])`,
                      donde:
                        - t (float): Tiempo actual.
                        - r (np.array): Vector de posición actual [x, y].
                        - v (np.array): Vector de velocidad actual [vx, vy].
                      La función debe devolver el vector de aceleración [ax, ay].
        
        t0 (float): Tiempo inicial.
        r0 (tuple): Posición inicial en 2D (x0, y0).
        v0 (tuple): Velocidad inicial en 2D (vx0, vy0).
        tf (float): Tiempo final de la simulación.
        h (float, optional): Tamaño del paso de integración. Por defecto es 0.001.

    Returns:
        tuple: 
            - t (np.array): Vector de tiempos en el intervalo [t0, tf] con paso `h`.
            - r (np.array): Matriz de posiciones [x, y] en cada paso de tiempo.
            - v (np.array): Matriz de velocidades [vx, vy] en cada paso de tiempo.

    Ejemplo de uso:
        >>> def fuerza(t, r, v):
        >>>     return np.array([0, -9.81])  # Fuerza gravitacional en el eje y
        >>> t0 = 0.0
        >>> r0 = (0, 0)  # Posición inicial
        >>> v0 = (10, 10)  # Velocidad inicial
        >>> tf = 2.0
        >>> t, r, v = rk4_method_second_order(f=fuerza, t0=t0, r0=r0, v0=v0, tf=tf, h=0.01)
        >>> plt.plot(r[:, 0], r[:, 1])
        >>> plt.xlabel("Posición X")
        >>> plt.ylabel("Posición Y")
        >>> plt.title("Trayectoria usando RK4")
        >>> plt.show()

    Notas:
        - El método RK4 es más preciso que RK2 y el método de Euler, y es ampliamente utilizado
          en la resolución numérica de ecuaciones diferenciales debido a su balance entre precisión y complejidad computacional.
        - El método es adecuado para sistemas donde el campo de fuerzas o aceleraciones es suave
          y no varía bruscamente en cada paso de tiempo.
    """


    # Inicializamos vectores para guardar la solución
    t = np.arange(start=t0, step=h, stop=tf + h)
    r = np.zeros(shape=(len(t), 2))
    v = np.zeros_like(r)


    # Fijamos las condiciones iniciales
    r[0] = r0
    v[0] = v0
    
    # Iteramos
    for n in range(len(t) - 1):

        # Cálculo de k1 y m1
        k1 = h * f(t[n], r[n], v[n])
        m1 = h * v[n]

        # Cálculo de k2 y m2
        k2 = h * f(t[n] + h/2, r[n] + m1/2, v[n] + k1/2)
        m2 = h * (v[n] + k1/2)

        # Cálculo de k3 y m3
        k3 = h * f(t[n] + h/2, r[n] + m2/2, v[n] + k2/2)
        m3 = h * (v[n] + k2/2)

        # Cálculo de k4 y m4
        k4 = h * f(t[n] + h, r[n] + m3, v[n] + k3)
        m4 = h * (v[n] + k3)

        # Actualizamos los valores de velocidad y posición
        v[n+1] = v[n] + (k1 + 2*k2 + 2*k3 + k4) / 6
        r[n+1] = r[n] + (m1 + 2*m2 + 2*m3 + m4) / 6

    return t, r, v

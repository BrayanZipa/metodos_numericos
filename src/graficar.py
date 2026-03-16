import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

"""
Genera la gráfica de una función y marca la ubicación de la raíz encontrada.
f_expr: expresión de sympy
x: símbolo de la variable (x)
raiz: valor numérico de la raíz aproximada
a, b: límites opcionales del intervalo para definir el rango de la gráfica.
"""
def graficarMetodos(f_expr, x, raiz, a=None, b=None):

    f_np = sp.lambdify(x, f_expr, 'numpy')
    
    # Determinar rango de la gráfica
    if a is not None and b is not None:
        margin = max(abs(b - a) * 0.5, 1.0)
        x_vals = np.linspace(a - margin, b + margin, 400)
    else:
        x_vals = np.linspace(raiz - 2, raiz + 2, 400)

    y_vals = f_np(x_vals)

    # plt.figure(figsize=(10, 6))
    plt.figure()
    plt.get_current_fig_manager().window.state('zoomed')
    plt.plot(x_vals, y_vals, label=f"f(x) = {f_expr}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter([raiz], [f_np(raiz)], color='red', label=f'Raíz aprox: {raiz:.6f}')
    
    plt.title("Gráfica de la función y su raíz")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

"""
Grafica la función original comparándola con el polinomio generado.
f_expr: expresión de sympy
p_expr: expresión del polinomio calculado
x: símbolo de la variable (x)
x0: punto de evaluación (centro) del polinomio
a, b: límites opcionales para el visualización en el eje x
"""
def graficarInterpolacion(f_expr, p_expr, x, x0, a=None, b=None):

    f_np = sp.lambdify(x, f_expr, 'numpy')
    p_np = sp.lambdify(x, p_expr, 'numpy')
    
    # Determinar rango de la gráfica
    if a is not None and b is not None:
        margin = max(abs(b - a) * 0.1, 0.5)
        x_vals = np.linspace(a - margin, b + margin, 400)
    else:
        x_vals = np.linspace(x0 - 2, x0 + 2, 400)

    y_vals_f = f_np(x_vals)
    y_vals_p = p_np(x_vals)

    # plt.figure(figsize=(10, 6))
    plt.figure()
    plt.get_current_fig_manager().window.state('zoomed')
    plt.plot(x_vals, y_vals_f, label=f"f(x) = {f_expr}", color='blue')
    plt.plot(x_vals, y_vals_p, label=f"P(x) (Polinomio de Taylor)", linestyle='--', color='orange')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter([x0], [float(f_np(x0))], color='red', label=f'Centro: x0={x0}')
    
    plt.title("Interpolación por Polinomio de Taylor")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

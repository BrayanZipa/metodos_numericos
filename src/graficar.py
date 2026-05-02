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
        margin = abs(b - a) * 0.1 if abs(b - a) > 0 else 1.0
        x_vals = np.linspace(a - margin, b + margin, 400)
    else:
        # Rango basado en la raíz
        x_vals = np.linspace(raiz - 2, raiz + 2, 400)

    y_vals = f_np(x_vals)

    # plt.figure(figsize=(10, 6))
    plt.figure()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except:
        pass
    plt.plot(x_vals, y_vals, label=f"f(x) = {f_expr}")
    
    # Solo mostrar ejes si están cerca del rango de datos
    if np.min(y_vals) <= 0 <= np.max(y_vals):
        plt.axhline(0, color='black', linewidth=0.5)
    if x_vals[0] <= 0 <= x_vals[-1]:
        plt.axvline(0, color='black', linewidth=0.5)
        
    plt.scatter([raiz], [f_np(raiz)], color='red', label=f'Raíz aprox: {raiz:.6f}')
    
    plt.title("Gráfica de la función y su raíz")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
Grafica la función original comparándola con el polinomio generado.
f_expr: expresión de sympy
p_expr: expresión del polinomio calculado
x: símbolo de la variable (x)
x0: punto de evaluación (centro) del polinomio
a, b: límites opcionales para el visualización en el eje x
"""
def graficarInterpolacion(f_expr, p_expr, x, x0=None, puntos_x=None, puntos_y=None, metodo="Taylor", a=None, b=None):
    # Convertir a funciones de numpy para graficar
    f_np = sp.lambdify(x, f_expr, 'numpy') if f_expr is not None else None
    p_np = sp.lambdify(x, p_expr, 'numpy')
    
    # Determinar el rango de valores de x para la evaluación de la función
    if a is not None and b is not None:
        margin = abs(b - a) * 0.1 if abs(b - a) > 0 else 0.5
        x_vals = np.linspace(a - margin, b + margin, 400)
    elif puntos_x is not None and len(puntos_x) > 0:
        x_min, x_max = min(puntos_x), max(puntos_x)
        range_x = x_max - x_min
        margin = range_x * 0.1 if range_x > 0 else 1.0
        x_vals = np.linspace(x_min - margin, x_max + margin, 400)
    elif x0 is not None:
        x_vals = np.linspace(x0 - 2, x0 + 2, 400)
    else:
        x_vals = np.linspace(-10, 10, 400)

    plt.figure()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except:
        pass

    # Graficar función original si existe
    if f_np is not None:
        try:
            y_vals_f = f_np(x_vals)
            plt.plot(x_vals, y_vals_f, label=f"f(x) = {f_expr}", color='blue', alpha=0.7)
        except Exception:
            pass # Evitar errores si la función no es evaluable en el rango

    # Graficar polinomio de interpolación
    y_vals_p = p_np(x_vals)
    plt.plot(x_vals, y_vals_p, label=f"P(x) (Polinomio de {metodo})", linestyle='--', color='orange', linewidth=2)
    
    # Solo mostrar ejes si están cerca del rango de datos para un mejor zoom
    y_min, y_max = np.min(y_vals_p), np.max(y_vals_p)
    if f_np is not None:
        try:
            y_vals_f = f_np(x_vals)
            y_min = min(y_min, np.min(y_vals_f))
            y_max = max(y_max, np.max(y_vals_f))
        except:
            pass

    if y_min <= 0 <= y_max:
        plt.axhline(0, color='black', linewidth=0.5)
    if x_vals[0] <= 0 <= x_vals[-1]:
        plt.axvline(0, color='black', linewidth=0.5)
    
    # Marcar puntos en la gráfica
    if metodo == "Taylor" and x0 is not None:
        try:
            y0 = float(f_expr.subs(x, x0)) if f_expr is not None else float(p_expr.subs(x, x0))
            plt.scatter([x0], [y0], color='red', zorder=5, label=f'Centro: x0={x0}')
        except:
            pass
    elif puntos_x is not None and puntos_y is not None:
        plt.scatter(puntos_x, puntos_y, color='red', zorder=5, label='Puntos originales')
    
    plt.title(f"Interpolación por Polinomio de {metodo}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
Grafica la función y el área calculada por métodos de integración.
f_expr: expresión de sympy
x: símbolo de la variable (x)
a, b: límites de integración
puntos_x, puntos_y: puntos evaluados por el método (para dibujar los trapecios u otros)
metodo: nombre del método de integración
"""
def graficarIntegracion(f_expr, x, a, b, puntos_x, puntos_y, metodo="Regla del Trapecio"):
    f_np = sp.lambdify(x, f_expr, 'numpy')
    
    # Determinar rango de la gráfica
    margin = abs(b - a) * 0.2 if abs(b - a) > 0 else 1.0
    x_vals = np.linspace(a - margin, b + margin, 400)
    
    try:
        y_vals = f_np(x_vals)
    except Exception:
        y_vals = np.zeros_like(x_vals) # Fallback si falla
        
    plt.figure()
    try:
        plt.get_current_fig_manager().window.state('zoomed')
    except:
        pass
        
    # Función original
    plt.plot(x_vals, y_vals, label=f"f(x) = {f_expr}", color='blue')
    
    # Rellenar el área de integración (aproximación)
    plt.fill_between(puntos_x, 0, puntos_y, color='orange', alpha=0.3, label='Área de integración')
    
    # Dibujar las líneas de los trapecios (bordes)
    for px, py in zip(puntos_x, puntos_y):
        plt.plot([px, px], [0, py], color='red', linestyle='--', linewidth=1)
    
    # Puntos evaluados
    plt.plot(puntos_x, puntos_y, color='red', marker='o', label='Puntos evaluados', markersize=4)
    
    # Ejes
    plt.axhline(0, color='black', linewidth=0.5)
    if x_vals[0] <= 0 <= x_vals[-1]:
        plt.axvline(0, color='black', linewidth=0.5)
        
    plt.title(f"Integración Numérica: {metodo}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

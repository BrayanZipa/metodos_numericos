import sympy as sp
import numpy as np

"""
Implementación del método de Punto Fijo para hallar raíces.
g_expr: expresión de la función g(x) tal que x = g(x)
x: símbolo de la variable (x)
x0: valor inicial
tol: máximo error permitido
max_iter: número máximo de iteraciones
"""
def puntoFijo(g_expr, x, x0, tol, max_iter = 100):
    g = sp.lambdify(x, g_expr, 'numpy')

    # Criterio de convergencia: |g'(x0)| < 1
    try:
        g_derivada = sp.diff(g_expr, x)
        valor_derivada = abs(float(g_derivada.subs(x, x0)))
        if valor_derivada >= 1:
            return None, f"Advertencia: El método puede no converger. |g'({x0})| = {valor_derivada:.4f} >= 1"
    except Exception as e:
        # En caso de error, se continua y se intenta generar las iteraciones
        pass

    iteraciones = []
    xi = x0
    
    for i in range(1, max_iter + 1):
        try:
            xi_siguiente = g(xi)
            if np.isnan(xi_siguiente) or np.isinf(xi_siguiente):
                return None, (
                    f"Iteración fuera del dominio.\n"
                    f"g({xi}) produjo un valor inválido."
            )
        except Exception as e:
            return None, f"Error al evaluar g(x): {e}"

        error = abs((xi_siguiente - xi) / xi_siguiente) if xi_siguiente != 0 else abs(xi_siguiente - xi)
        
        iteraciones.append({
            'iter': i,
            'xi': xi,
            'g(xi)': xi_siguiente,
            'error': error
        })

        if error < tol:
            break
        
        xi = xi_siguiente

    return iteraciones, None
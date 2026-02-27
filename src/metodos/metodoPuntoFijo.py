import sympy as sp

"""
Implementación del método de Punto Fijo para hallar raíces.
g_expr: expresión de la función g(x) tal que x = g(x)
x_orig: símbolo de la variable (x)
x0: valor inicial
tol: tolerancia
max_iter: número máximo de iteraciones
"""
def puntoFijo(g_expr, x_orig, x0, tol, max_iter=100):
    g = sp.lambdify(x_orig, g_expr, 'numpy')

    iteraciones = []
    xi = x0
    
    for i in range(1, max_iter + 1):
        try:
            xi_siguiente = g(xi)
        except Exception as e:
            return None, f"Error al evaluar g(x): {e}"

        error = abs(xi_siguiente - xi)
        
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
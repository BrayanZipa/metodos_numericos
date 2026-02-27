import sympy as sp

"""
Implementación del método de la Secante para hallar raíces.
f_expr: expresión de sympy
x_orig: símbolo de la variable (x)
x0, x1: valores iniciales
tol: tolerancia
max_iter: número máximo de iteraciones
"""
def secante(f_expr, x_orig, x0, x1, tol, max_iter=100):
    f = sp.lambdify(x_orig, f_expr, 'numpy')

    iteraciones = []
    
    for i in range(1, max_iter + 1):
        f_x0 = f(x0)
        f_x1 = f(x1)

        if abs(f_x1 - f_x0) < 1e-12:
            return None, "Error: División por cero en el método de la secante."

        # xi+1 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x_nuevo = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs(x_nuevo - x1)
        
        iteraciones.append({
            'iter': i,
            'x0': x0,
            'x1': x1,
            'f(x0)': f_x0,
            'f(x1)': f_x1,
            'xi+1': x_nuevo,
            'error': error
        })

        if error < tol or abs(f_x1) < tol:
            break
        
        x0 = x1
        x1 = x_nuevo

    return iteraciones, None
import sympy as sp

"""
Implementación del método de Newton-Raphson para hallar raíces.
f_expr: expresión de sympy
x_orig: símbolo de la variable (x)
x0: valor inicial
tol: tolerancia
max_iter: número máximo de iteraciones
"""
def newtonRaphson(f_expr, x_orig, x0, tol, max_iter=100):
    df_expr = sp.diff(f_expr, x_orig)
    f = sp.lambdify(x_orig, f_expr, 'numpy')
    df = sp.lambdify(x_orig, df_expr, 'numpy')

    iteraciones = []
    xi = x0
    
    for i in range(1, max_iter + 1):
        f_xi = f(xi)
        df_xi = df(xi)

        if abs(df_xi) < 1e-12:
            return None, "Error: Derivada cercana a cero. El método no puede continuar."

        xi_nuevo = xi - f_xi / df_xi
        error = abs(xi_nuevo - xi)
        
        iteraciones.append({
            'iter': i,
            'xi': xi,
            'f(xi)': f_xi,
            'df(xi)': df_xi,
            'xi+1': xi_nuevo,
            'error': error
        })

        if error < tol or abs(f_xi) < tol:
            break
        
        xi = xi_nuevo

    return iteraciones, None

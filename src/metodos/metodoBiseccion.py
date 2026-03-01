import sympy as sp

"""
Implementación del método de bisección para hallar raíces.
f_expr: expresión de sympy
x: símbolo de la variable (x)
a, b: intervalo inicial
tol: máximo error permitido
max_iter: número máximo de iteraciones
"""
def biseccion(f_expr, x, a, b, tol, max_iter = 100):
    f_eval = sp.lambdify(x, f_expr, 'numpy')

    if f_eval(a) * f_eval(b) >= 0:
        return None, "El método de bisección no garantiza una raíz en el intervalo ingresado (f(a) * f(b) >= 0)."

    iteraciones = []
    pn = a
    for i in range(1, max_iter + 1):
        pn_anterior = pn
        pn = (a + b) / 2
        f_a = f_eval(a)
        f_pn = f_eval(pn)
        prod = f_a * f_pn
        
        error = abs((pn - pn_anterior) / pn) if i > 1 else abs(b - a)
        
        iteraciones.append({
            'iter': i,
            'a': a,
            'b': b,
            'pn': pn,
            'f(a)': f_a,
            'f(pn)': f_pn,
            'f(a)*f(pn)': prod,
            'error': error
        })

        if f_pn == 0 or error < tol:
            break

        if prod < 0:
            b = pn
        else:
            a = pn

    return iteraciones, None

import sympy as sp

"""
Resuelve una Ecuación Diferencial Ordinaria de primer orden y' = f(x, y) usando el método de Euler.
f_expr: expresión de sympy de la derivada f(x, y).
x, y: símbolos de las variables independiente y dependiente.
x0, y0: valores iniciales (condición inicial).
xf: punto final de evaluación.
h: tamaño de paso.
exact_expr: expresión de la solución exacta (opcional para cálculo de error).
"""
def euler(f_expr, x, y, x0, y0, xf, h, exact_expr=None):
    n = int(round((xf - x0) / h))
    
    xi = x0
    yi = y0
    
    tabla = []
    
    for i in range(n + 1):
        # Fórmula de Euler: y_{i+1} = yi + h * f(xi, yi)
        dy = float(f_expr.subs({x: xi, y: yi}))
        yi_next = yi + h * dy

        if exact_expr is not None:
            valor_real = float(exact_expr.subs({x: xi}))
            error_abs = abs(valor_real - yi)
            if valor_real != 0:
                error_rel = error_abs / abs(valor_real)
            else:
                error_rel = 0.0
            
            tabla.append({
                'n': i,
                'xi': xi,
                'yi': yi,
                'f(xi, yi)': dy,
                'yi+1': yi_next,
                'valor_real': valor_real,
                'error_abs': error_abs,
                'error_rel': error_rel
            })
        else:
            tabla.append({
                'n': i,
                'xi': xi,
                'yi': yi,
                'f(xi, yi)': dy,
                'yi+1': yi_next
            })
            
        if i < n:
            yi = yi_next
            xi = xi + h
            
    return tabla
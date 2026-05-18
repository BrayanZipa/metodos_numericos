import sympy as sp

"""
Resuelve una Ecuación Diferencial Ordinaria de primer orden y' = f(x, y) usando el método de Runge-Kutta de 2do orden.
f_expr: expresión de sympy de la derivada f(x, y).
x, y: símbolos de las variables independiente y dependiente.
x0, y0: valores iniciales (condición inicial).
xf: punto final de evaluación.
h: tamaño de paso.
exact_expr: expresión de la solución exacta (opcional para cálculo de error).
"""
def rungeKuttaSegundoOrden(f_expr, x, y, x0, y0, xf, h, exact_expr=None):
    n = int(round((xf - x0) / h))
    
    xi = x0
    yi = y0
    
    tabla = []
    
    for i in range(n + 1):
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
                'valor_real': valor_real,
                'error_abs': error_abs,
                'error_rel': error_rel
            })
        else:
            tabla.append({
                'n': i,
                'xi': xi,
                'yi': yi
            })
            
        # Cálculos de Runge-Kutta de 2do orden (Punto medio)
        # Fórmulas: k1 = h*f(xi, yi), k2 = h*f(xi + h/2, yi + k1/2), yi+1 = yi + k2
        f_xy = float(f_expr.subs({x: xi, y: yi}))
        k1 = h * f_xy
        k2 = h * float(f_expr.subs({x: xi + h/2, y: yi + k1/2}))
        yi_next = yi + k2
        xi_next = xi + h

        # Cálculos de Runge-Kutta de 2do orden (Método de Heun)
        # Fórmulas: k1 = h*f(xi, yi), k2 = h*f(xi + h, yi + k1), yi+1 = yi + (1/2)*(k1 + k2)
        # f_xy = float(f_expr.subs({x: xi, y: yi}))
        # k1 = h * f_xy
        # k2 = h * float(f_expr.subs({x: xi + h, y: yi + k1}))
        # yi_next = yi + (1/2) * (k1 + k2)
        # xi_next = xi + h

        # Cálculos de Runge-Kutta de 2do orden (Ralston)
        # Fórmulas: k1 = h*f(xi, yi), k2 = h*f(xi + 3h/4, yi + 3k1/4), yi+1 = yi + (1/3)k1 + (2/3)k2
        # f_xy = float(f_expr.subs({x: xi, y: yi}))
        # k1 = h * f_xy
        # k2 = h * float(f_expr.subs({x: xi + 3*h/4, y: yi + 3*k1/4}))
        # yi_next = yi + (1/3)*k1 + (2/3)*k2
        # xi_next = xi + h
        
        tabla[-1]['f(xi, yi)'] = f_xy
        tabla[-1]['k1'] = k1
        tabla[-1]['k2'] = k2
        tabla[-1]['yi+1'] = yi_next
        
        yi = yi_next
        xi = xi_next
            
    return tabla

"""
Resuelve una Ecuación Diferencial Ordinaria de primer orden y' = f(x, y) usando el método de Runge-Kutta de 4to orden.
f_expr: expresión de sympy de la derivada f(x, y).
x, y: símbolos de las variables independiente y dependiente.
x0, y0: valores iniciales (condición inicial).
xf: punto final de evaluación.
h: tamaño de paso.
exact_expr: expresión de la solución exacta (opcional para cálculo de error).
"""
def rungeKuttaCuartoOrden(f_expr, x, y, x0, y0, xf, h, exact_expr=None):
    n = int(round((xf - x0) / h))
    
    xi = x0
    yi = y0
    
    tabla = []
    
    for i in range(n + 1):
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
                'valor_real': valor_real,
                'error_abs': error_abs,
                'error_rel': error_rel
            })
        else:
            tabla.append({
                'n': i,
                'xi': xi,
                'yi': yi
            })
            
        # Cálculos de Runge-Kutta de 4to orden
        # Fórmulas: k1 = h*f(xi, yi), k2 = h*f(xi + h/2, yi + k1/2), k3 = h*f(xi + h/2, yi + k2/2), k4 = h*f(xi + h, yi + k3)
        # yi+1 = yi + (1/6)*(k1 + 2k2 + 2k3 + k4)
        f_xy = float(f_expr.subs({x: xi, y: yi}))
        k1 = h * f_xy
        k2 = h * float(f_expr.subs({x: xi + h/2, y: yi + k1/2}))
        k3 = h * float(f_expr.subs({x: xi + h/2, y: yi + k2/2}))
        k4 = h * float(f_expr.subs({x: xi + h, y: yi + k3}))
        
        yi_next = yi + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        xi_next = xi + h
        
        tabla[-1]['f(xi, yi)'] = f_xy
        tabla[-1]['k1'] = k1
        tabla[-1]['k2'] = k2
        tabla[-1]['k3'] = k3
        tabla[-1]['k4'] = k4
        tabla[-1]['yi+1'] = yi_next
        
        yi = yi_next
        xi = xi_next
            
    return tabla

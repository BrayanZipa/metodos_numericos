import sympy as sp
import math

"""
Calcula la integral numérica de una función usando la Regla del Trapecio Simple.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
"""
def reglaTrapecioSimple(f_expr, x, a, b):
    try:
        h = b - a
        fa = float(f_expr.subs(x, a))
        fb = float(f_expr.subs(x, b))
        
        # Calcular integral valor aproximado
        # Fórmula: I ≈ (h/2) * [f(a) + f(b)]
        valor_aprox = (h / 2) * (fa + fb)
        
        # Calcular integral valor exacto
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        tabla = [
            {'i': 0, 'x': a, 'f(x)': fa},
            {'i': 1, 'x': b, 'f(x)': fb}
        ]
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Trapecio Simple: {e}"

"""
Calcula la integral numérica de una función usando la Regla del Trapecio Compuesta.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
n: número de subintervalos.
"""
def reglaTrapecioCompuesta(f_expr, x, a, b, n):
    try:
        if n < 1:
            return None, None, None, None, None, "El número de subintervalos (n) debe ser mayor o igual a 1."
            
        h = (b - a) / n
        tabla = []
        
        # Evaluar en a
        fa = float(f_expr.subs(x, a))
        tabla.append({'i': 0, 'x': a, 'f(x)': fa})
        
        # Puntos interiores
        suma_interior = 0
        for i in range(1, n):
            xi = a + i * h
            fxi = float(f_expr.subs(x, xi))
            suma_interior += fxi
            tabla.append({'i': i, 'x': xi, 'f(x)': fxi})
            
        # Evaluar en b
        fb = float(f_expr.subs(x, b))
        tabla.append({'i': n, 'x': b, 'f(x)': fb})
        
        # Calcular integral valor aproximado
        # Fórmula: I ≈ (h/2) * [f(a) + 2*Σf(xi) + f(b)]
        valor_aprox = (h / 2) * (fa + 2 * suma_interior + fb)
        
        # Calcular integral valor exacto
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Trapecio Compuesta: {e}"
import sympy as sp

"""
Calcula la integral numérica de una función usando la Regla de Simpson 1/3 Simple.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
"""
def simpsonUnTercioSimple(f_expr, x, a, b):
    try:
        h = (b - a) / 2
        x0 = a
        x1 = a + h
        x2 = b
        
        f0 = float(f_expr.subs(x, x0))
        f1 = float(f_expr.subs(x, x1))
        f2 = float(f_expr.subs(x, x2))
        
        # Fórmula: I ≈ (h/3) * [f(x0) + 4f(x1) + f(x2)]
        valor_aprox = (h / 3) * (f0 + 4 * f1 + f2)
        
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        tabla = [
            {'i': 0, 'x': x0, 'f(x)': f0},
            {'i': 1, 'x': x1, 'f(x)': f1},
            {'i': 2, 'x': x2, 'f(x)': f2}
        ]
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Simpson 1/3 Simple: {e}"

"""
Calcula la integral numérica de una función usando la Regla de Simpson 1/3 Compuesta.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
n: número de subintervalos (debe ser un número par).
"""
def simpsonUnTercioCompuesta(f_expr, x, a, b, n):
    try:
        if n < 2 or n % 2 != 0:
            return None, None, None, None, None, "Para Simpson 1/3, el número de subintervalos (n) debe ser par y mayor o igual a 2."
            
        h = (b - a) / n
        tabla = []
        
        suma_impares = 0
        suma_pares = 0
        
        f0 = float(f_expr.subs(x, a))
        tabla.append({'i': 0, 'x': a, 'f(x)': f0})
        
        xi = a
        for i in range(1, n):
            xi = a + i * h
            fxi = float(f_expr.subs(x, xi))
            if i % 2 == 0:
                suma_pares += fxi
            else:
                suma_impares += fxi
            tabla.append({'i': i, 'x': xi, 'f(x)': fxi})
            
        fn = float(f_expr.subs(x, b))
        tabla.append({'i': n, 'x': b, 'f(x)': fn})
        
        # Fórmula: I ≈ (h/3) * [f(x0) + 4*Σf(x_impares) + 2*Σf(x_pares) + f(xn)]
        valor_aprox = (h / 3) * (f0 + 4 * suma_impares + 2 * suma_pares + fn)
        
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Simpson 1/3 Compuesta: {e}"

"""
Calcula la integral numérica de una función usando la Regla de Simpson 3/8 Simple.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
"""
def simpsonTresOctavosSimple(f_expr, x, a, b):
    try:
        h = (b - a) / 3
        x0 = a
        x1 = a + h
        x2 = a + 2 * h
        x3 = b
        
        f0 = float(f_expr.subs(x, x0))
        f1 = float(f_expr.subs(x, x1))
        f2 = float(f_expr.subs(x, x2))
        f3 = float(f_expr.subs(x, x3))
        
        # Fórmula: I ≈ (3h/8) * [f(x0) + 3f(x1) + 3f(x2) + f(x3)]
        valor_aprox = (3 * h / 8) * (f0 + 3 * f1 + 3 * f2 + f3)
        
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        tabla = [
            {'i': 0, 'x': x0, 'f(x)': f0},
            {'i': 1, 'x': x1, 'f(x)': f1},
            {'i': 2, 'x': x2, 'f(x)': f2},
            {'i': 3, 'x': x3, 'f(x)': f3}
        ]
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Simpson 3/8 Simple: {e}"

"""
Calcula la integral numérica de una función usando la Regla de Simpson 3/8 Compuesta.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
n: número de subintervalos (debe ser múltiplo de 3).
"""
def simpsonTresOctavosCompuesta(f_expr, x, a, b, n):
    try:
        if n < 3 or n % 3 != 0:
            return None, None, None, None, None, "Para Simpson 3/8, el número de subintervalos (n) debe ser múltiplo de 3 y mayor o igual a 3."
            
        h = (b - a) / n
        tabla = []
        
        suma_multiplos_3 = 0
        suma_resto = 0
        
        f0 = float(f_expr.subs(x, a))
        tabla.append({'i': 0, 'x': a, 'f(x)': f0})
        
        xi = a
        for i in range(1, n):
            xi = a + i * h
            fxi = float(f_expr.subs(x, xi))
            if i % 3 == 0:
                suma_multiplos_3 += fxi
            else:
                suma_resto += fxi
            tabla.append({'i': i, 'x': xi, 'f(x)': fxi})
            
        fn = float(f_expr.subs(x, b))
        tabla.append({'i': n, 'x': b, 'f(x)': fn})
        
        # Fórmula: I ≈ (3h/8) * [f(x0) + 3*Σf(resto) + 2*Σf(multiplos de 3) + f(xn)]
        valor_aprox = (3 * h / 8) * (f0 + 3 * suma_resto + 2 * suma_multiplos_3 + fn)
        
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Simpson 3/8 Compuesta: {e}"
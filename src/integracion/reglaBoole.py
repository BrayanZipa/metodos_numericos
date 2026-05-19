import sympy as sp

"""
Calcula la integral numérica de una función usando la Regla de Boole Simple.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
"""
def booleSimple(f_expr, x, a, b):
    try:
        h = (b - a) / 4
        x0 = a
        x1 = a + h
        x2 = a + 2 * h
        x3 = a + 3 * h
        x4 = b
        
        f0 = float(f_expr.subs(x, x0))
        f1 = float(f_expr.subs(x, x1))
        f2 = float(f_expr.subs(x, x2))
        f3 = float(f_expr.subs(x, x3))
        f4 = float(f_expr.subs(x, x4))
        
        # Fórmula: I ≈ (2h/45) * [7f(x0) + 32f(x1) + 12f(x2) + 32f(x3) + 7f(x4)]
        valor_aprox = (2 * h / 45) * (7 * f0 + 32 * f1 + 12 * f2 + 32 * f3 + 7 * f4)
        
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        tabla = [
            {'i': 0, 'x': x0, 'f(x)': f0},
            {'i': 1, 'x': x1, 'f(x)': f1},
            {'i': 2, 'x': x2, 'f(x)': f2},
            {'i': 3, 'x': x3, 'f(x)': f3},
            {'i': 4, 'x': x4, 'f(x)': f4}
        ]
        
        return valor_aprox, valor_exacto, error_abs, tabla, h, None
    except Exception as e:
        return None, None, None, None, None, f"Error en Boole Simple: {e}"

"""
Calcula la integral numérica de una función usando la Regla de Boole Compuesta.
f_expr: función simbólica a integrar.
x: símbolo de la variable de integración.
a: límite inferior de integración.
b: límite superior de integración.
n: número de subintervalos (debe ser múltiplo de 4).
"""
def booleCompuesta(f_expr, x, a, b, n):
    try:
        if n < 4 or n % 4 != 0:
            return None, None, None, None, None, "Para Boole, el número de subintervalos (n) debe ser múltiplo de 4 y mayor o igual a 4."
            
        h = (b - a) / n
        tabla = []
        
        suma_impares = 0
        suma_pares_no_m4 = 0
        suma_m4 = 0
        
        f0 = float(f_expr.subs(x, a))
        tabla.append({'i': 0, 'x': a, 'f(x)': f0})
        
        xi = a
        for i in range(1, n):
            xi = a + i * h
            fxi = float(f_expr.subs(x, xi))
            if i % 2 != 0:
                suma_impares += fxi
            elif i % 4 == 0:
                suma_m4 += fxi
            else:
                suma_pares_no_m4 += fxi
            tabla.append({'i': i, 'x': xi, 'f(x)': fxi})
            
        fn = float(f_expr.subs(x, b))
        tabla.append({'i': n, 'x': b, 'f(x)': fn})
        
        # Fórmula: I ≈ (2h/45) * [7f(x0) + 32*Σf(x_impares) + 12*Σf(x_pares_no_m4) + 14*Σf(x_m4) + 7f(xn)]
        valor_aprox = (2 * h / 45) * (7 * f0 + 32 * suma_impares + 12 * suma_pares_no_m4 + 14 * suma_m4 + 7 * fn)
        
        integral_exacta = sp.integrate(f_expr, (x, a, b))
        valor_exacto = float(integral_exacta)
        
        error_abs = abs(valor_exacto - valor_aprox)
        
        # Calcular áreas por subintervalo
        tabla_areas = []
        for i in range(0, n, 4):
            xi = tabla[i]['x']
            xi_4 = tabla[i+4]['x']
            fxi = tabla[i]['f(x)']
            fxi_1 = tabla[i+1]['f(x)']
            fxi_2 = tabla[i+2]['f(x)']
            fxi_3 = tabla[i+3]['f(x)']
            fxi_4 = tabla[i+4]['f(x)']
            area = (2 * h / 45) * (7 * fxi + 32 * fxi_1 + 12 * fxi_2 + 32 * fxi_3 + 7 * fxi_4)
            tabla_areas.append({'i_inicio': i, 'i_fin': i+4, 'xi': xi, 'xi_fin': xi_4, 'area': area})
            
        return valor_aprox, valor_exacto, error_abs, tabla, tabla_areas, h, None
    except Exception as e:
        return None, None, None, None, None, None, f"Error en Boole Compuesta: {e}"
    
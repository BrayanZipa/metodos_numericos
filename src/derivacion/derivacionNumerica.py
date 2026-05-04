import sympy as sp
import math

"""
Calcula la derivada numérica de una función en un punto x0 usando fórmulas de 2 puntos.
f_expr: función simbólica
x: símbolo de la variable (x)
x0: punto donde se evalúa la derivada
h: tamaño de paso
tipo_diferencia: 1 (Progresiva), 2 (Regresiva), 3 (Central)
"""
def derivadaDosPuntos(f_expr, x, x0, h, tipo_diferencia):
    try:
        tabla = []
        error_teorico = None
        
        if tipo_diferencia == 1:
            nombre_regla = "Diferencia progresiva de 2 puntos"
            # Diferencia progresiva fórmula: D ≈ (f(x0+h) - f(x0)) / h
            fx0 = float(f_expr.subs(x, x0))
            fx1 = float(f_expr.subs(x, x0 + h))
            valor_aprox = (fx1 - fx0) / h
            
            tabla = [
                {'i': 0, 'x': x0, 'f(x)': fx0},
                {'i': 1, 'x': x0 + h, 'f(x)': fx1}
            ]
            
            # Error teórico progresiva: |(h/2) * f''(x0)|
            df2 = sp.diff(f_expr, x, 2)
            error_teorico = abs(float((h / 2) * df2.subs(x, x0)))
            
        elif tipo_diferencia == 2:
            nombre_regla = "Diferencia regresiva de 2 puntos"
            # Diferencia regresiva fórmula: D ≈ (f(x0) - f(x0-h)) / h
            fx0 = float(f_expr.subs(x, x0))
            fx_1 = float(f_expr.subs(x, x0 - h))
            valor_aprox = (fx0 - fx_1) / h
            
            tabla = [
                {'i': 0, 'x': x0 - h, 'f(x)': fx_1},
                {'i': 1, 'x': x0, 'f(x)': fx0}
            ]
            
            # Error teórico regresiva: |(h/2) * f''(x0)|
            df2 = sp.diff(f_expr, x, 2)
            error_teorico = abs(float((h / 2) * df2.subs(x, x0)))
            
        elif tipo_diferencia == 3:
            nombre_regla = "Diferencia central de 2 puntos"
            # Diferencia central fórmula: D ≈ (f(x0+h) - f(x0-h)) / (2h)
            fx1 = float(f_expr.subs(x, x0 + h))
            fx_1 = float(f_expr.subs(x, x0 - h))
            valor_aprox = (fx1 - fx_1) / (2 * h)
            
            tabla = [
                {'i': 0, 'x': x0 - h, 'f(x)': fx_1},
                {'i': 1, 'x': x0 + h, 'f(x)': fx1}
            ]
            
            # Error teórico central: |(h^2/6) * f'''(x0)|
            df3 = sp.diff(f_expr, x, 3)
            error_teorico = abs(float(((h**2) / 6) * df3.subs(x, x0)))
        else:
            return None, None, None, None, None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central)."
            
        # Calcular valor exacto usando sympy
        df_expr = sp.diff(f_expr, x)
        valor_exacto = float(df_expr.subs(x, x0))
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), float(error_teorico), tabla, h, nombre_regla, None
        
    except Exception as e:
        return None, None, None, None, None, None, None, f"Error en el cálculo de la derivada: {e}"

"""
Calcula la primera derivada numérica de una función usando fórmulas de 3 puntos.
f_expr: función simbólica
x: símbolo de la variable (x)
x0: punto donde se evalúa la derivada
h: tamaño de paso
tipo_diferencia: 1 (Progresiva), 2 (Regresiva), 3 (Central)
    """
def derivadaTresPuntos(f_expr, x, x0, h, tipo_diferencia=1):
    try:
        tabla = []
        error_teorico = None
        
        if tipo_diferencia == 1:
            nombre_regla = "Diferencia progresiva de 3 puntos"
            # Diferencia progresiva fórmula: D ≈ (-3f(x0) + 4f(x0+h) - f(x0+2h)) / (2h)
            fx0 = float(f_expr.subs(x, x0))
            fx1 = float(f_expr.subs(x, x0 + h))
            fx2 = float(f_expr.subs(x, x0 + 2 * h))
            valor_aprox = (-3 * fx0 + 4 * fx1 - fx2) / (2 * h)
            
            tabla = [
                {'i': 0, 'x': x0, 'f(x)': fx0},
                {'i': 1, 'x': x0 + h, 'f(x)': fx1},
                {'i': 2, 'x': x0 + 2 * h, 'f(x)': fx2}
            ]
            
            # Error teórico: |(h^2/3) * f'''(x0)|
            df3 = sp.diff(f_expr, x, 3)
            error_teorico = abs(float(((h**2) / 3) * df3.subs(x, x0)))
            
        elif tipo_diferencia == 2:
            nombre_regla = "Diferencia regresiva de 3 puntos"
            # Diferencia regresiva fórmula: D ≈ (3f(x0) - 4f(x0-h) + f(x0-2h)) / (2h)
            fx0 = float(f_expr.subs(x, x0))
            fx_1 = float(f_expr.subs(x, x0 - h))
            fx_2 = float(f_expr.subs(x, x0 - 2 * h))
            valor_aprox = (3 * fx0 - 4 * fx_1 + fx_2) / (2 * h)
            
            tabla = [
                {'i': 0, 'x': x0 - 2 * h, 'f(x)': fx_2},
                {'i': 1, 'x': x0 - h, 'f(x)': fx_1},
                {'i': 2, 'x': x0, 'f(x)': fx0}
            ]
            
            # Error teórico: |(h^2/3) * f'''(x0)|
            df3 = sp.diff(f_expr, x, 3)
            error_teorico = abs(float(((h**2) / 3) * df3.subs(x, x0)))
            
        elif tipo_diferencia == 3:
            nombre_regla = "Diferencia central de 3 puntos"
            # Diferencia central fórmula: D ≈ (f(x0+h) - f(x0-h)) / (2h)
            fx1 = float(f_expr.subs(x, x0 + h))
            fx_1 = float(f_expr.subs(x, x0 - h))
            valor_aprox = (fx1 - fx_1) / (2 * h)
            
            tabla = [
                {'i': 0, 'x': x0 - h, 'f(x)': fx_1},
                {'i': 1, 'x': x0 + h, 'f(x)': fx1}
            ]
            
            # Error teórico: |(h^2/6) * f'''(x0)|
            df3 = sp.diff(f_expr, x, 3)
            error_teorico = abs(float(((h**2) / 6) * df3.subs(x, x0)))
        else:
            return None, None, None, None, None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central)."
            
        # Calcular valor exacto usando sympy (primera derivada)
        df_expr = sp.diff(f_expr, x, 1)
        valor_exacto = float(df_expr.subs(x, x0))
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), float(error_teorico), tabla, h, nombre_regla, None
        
    except Exception as e:
        return None, None, None, None, None, None, None, f"Error en el cálculo de la derivada: {e}"

"""
Calcula la primera derivada numérica de una función usando fórmulas de 5 puntos (O(h^4)).
f_expr: función simbólica
x: símbolo de la variable (x)
x0: punto donde se evalúa la derivada
h: tamaño de paso
tipo_diferencia: 1 (Progresiva), 2 (Regresiva), 3 (Central)
"""
def derivadaCincoPuntos(f_expr, x, x0, h, tipo_diferencia=1):
    try:
        tabla = []
        error_teorico = None
        
        if tipo_diferencia == 1:
            nombre_regla = "Diferencia progresiva de 5 puntos"
            # Diferencia progresiva fórmula: D ≈ (-25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h)) / 12h
            fx0 = float(f_expr.subs(x, x0))
            fx1 = float(f_expr.subs(x, x0 + h))
            fx2 = float(f_expr.subs(x, x0 + 2 * h))
            fx3 = float(f_expr.subs(x, x0 + 3 * h))
            fx4 = float(f_expr.subs(x, x0 + 4 * h))
            valor_aprox = (-25 * fx0 + 48 * fx1 - 36 * fx2 + 16 * fx3 - 3 * fx4) / (12 * h)
            
            tabla = [
                {'i': 0, 'x': x0, 'f(x)': fx0},
                {'i': 1, 'x': x0 + h, 'f(x)': fx1},
                {'i': 2, 'x': x0 + 2 * h, 'f(x)': fx2},
                {'i': 3, 'x': x0 + 3 * h, 'f(x)': fx3},
                {'i': 4, 'x': x0 + 4 * h, 'f(x)': fx4}
            ]
            
            # Error teórico: |(h^4/5) * f^(5)(x0)|
            df5 = sp.diff(f_expr, x, 5)
            error_teorico = abs(float(((h**4) / 5) * df5.subs(x, x0)))
            
        elif tipo_diferencia == 2:
            nombre_regla = "Diferencia regresiva de 5 puntos"
            # Diferencia regresiva fórmula: D ≈ (25f(x) - 48f(x-h) + 36f(x-2h) - 16f(x-3h) + 3f(x-4h)) / 12h
            fx0 = float(f_expr.subs(x, x0))
            fx_1 = float(f_expr.subs(x, x0 - h))
            fx_2 = float(f_expr.subs(x, x0 - 2 * h))
            fx_3 = float(f_expr.subs(x, x0 - 3 * h))
            fx_4 = float(f_expr.subs(x, x0 - 4 * h))
            valor_aprox = (25 * fx0 - 48 * fx_1 + 36 * fx_2 - 16 * fx_3 + 3 * fx_4) / (12 * h)
            
            tabla = [
                {'i': 0, 'x': x0 - 4 * h, 'f(x)': fx_4},
                {'i': 1, 'x': x0 - 3 * h, 'f(x)': fx_3},
                {'i': 2, 'x': x0 - 2 * h, 'f(x)': fx_2},
                {'i': 3, 'x': x0 - h, 'f(x)': fx_1},
                {'i': 4, 'x': x0, 'f(x)': fx0}
            ]
            
            # Error teórico: |(h^4/5) * f^(5)(x0)|
            df5 = sp.diff(f_expr, x, 5)
            error_teorico = abs(float(((h**4) / 5) * df5.subs(x, x0)))
            
        elif tipo_diferencia == 3:
            nombre_regla = "Diferencia central de 5 puntos"
            # Diferencia central fórmula: D ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / 12h
            fx_2 = float(f_expr.subs(x, x0 - 2 * h))
            fx_1 = float(f_expr.subs(x, x0 - h))
            fx1 = float(f_expr.subs(x, x0 + h))
            fx2 = float(f_expr.subs(x, x0 + 2 * h))
            valor_aprox = (fx_2 - 8 * fx_1 + 8 * fx1 - fx2) / (12 * h)
            
            tabla = [
                {'i': 0, 'x': x0 - 2 * h, 'f(x)': fx_2},
                {'i': 1, 'x': x0 - h, 'f(x)': fx_1},
                {'i': 2, 'x': x0 + h, 'f(x)': fx1},
                {'i': 3, 'x': x0 + 2 * h, 'f(x)': fx2}
            ]
            
            # Error teórico: |(h^4/30) * f^(5)(x0)|
            df5 = sp.diff(f_expr, x, 5)
            error_teorico = abs(float(((h**4) / 30) * df5.subs(x, x0)))
        else:
            return None, None, None, None, None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central)."
            
        # Calcular valor exacto usando sympy (primera derivada)
        df_expr = sp.diff(f_expr, x, 1)
        valor_exacto = float(df_expr.subs(x, x0))
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), float(error_teorico), tabla, h, nombre_regla, None
        
    except Exception as e:
        return None, None, None, None, None, None, None, f"Error en el cálculo de la derivada: {e}"

"""
Calcula la derivada numérica de orden n de una función en un punto x0.
f_expr: función simbólica
x: símbolo de la variable (x)
x0: punto donde se evalúa la derivada
h: tamaño de paso
orden: orden de la derivada (1 para primera derivada, 2 para segunda, etc.)
tipo_diferencia: 1 (Progresiva), 2 (Regresiva), 3 (Central)
"""
def derivadaOrdenSuperior(f_expr, x, x0, h, orden=1, tipo_diferencia=1):
    try:
        n = orden
        if n < 1:
            return None, None, None, None, None, None, "El orden de la derivada debe ser mayor o igual a 1."
            
        tabla = []
        valor_aprox = 0
        nombres = {1: "Diferencia progresiva", 2: "Diferencia regresiva", 3: "Diferencia central"}
        nombre_regla = f"Orden {n} - {nombres.get(tipo_diferencia, 'Desconocido')}"
        
        if tipo_diferencia == 1:
            # Diferencia progresiva fórmula: D ≈ (1/h^n) * Σ[(-1)^(n-k) * C(n,k) * f(x0+kh)] para k=0..n
            for k in range(n + 1):
                coef = ((-1)**(n - k)) * math.comb(n, k)
                xi = x0 + k * h
                fxi = float(f_expr.subs(x, xi))
                valor_aprox += coef * fxi
                tabla.append({'i': k, 'x': xi, 'f(x)': fxi})
            valor_aprox = valor_aprox / (h**n)
            
        elif tipo_diferencia == 2:
            # Diferencia regresiva fórmula: D ≈ (1/h^n) * Σ[(-1)^k * C(n,k) * f(x0-kh)] para k=0..n
            puntos_temp = []
            for k in range(n + 1):
                coef = ((-1)**k) * math.comb(n, k)
                xi = x0 - k * h
                fxi = float(f_expr.subs(x, xi))
                valor_aprox += coef * fxi
                puntos_temp.append({'x': xi, 'f(x)': fxi})
            valor_aprox = valor_aprox / (h**n)
            # Ordenar los puntos de menor a mayor x
            puntos_temp.sort(key=lambda p: p['x'])
            for i, p in enumerate(puntos_temp):
                tabla.append({'i': i, 'x': p['x'], 'f(x)': p['f(x)']})
            
        elif tipo_diferencia == 3:
            # Diferencia central fórmula: D ≈ (1/(2h)^n) * Σ[(-1)^k * C(n,k) * f(x0 + (n-2k)h)] para k=0..n
            puntos_temp = []
            for k in range(n + 1):
                coef = ((-1)**k) * math.comb(n, k)
                xi = x0 + (n - 2 * k) * h
                fxi = float(f_expr.subs(x, xi))
                valor_aprox += coef * fxi
                puntos_temp.append({'x': xi, 'f(x)': fxi})
            valor_aprox = valor_aprox / ((2 * h)**n)
            # Ordenar los puntos de menor a mayor x
            puntos_temp.sort(key=lambda p: p['x'])
            for i, p in enumerate(puntos_temp):
                tabla.append({'i': i, 'x': p['x'], 'f(x)': p['f(x)']})
            
        else:
            return None, None, None, None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central)."
            
        # Calcular valor exacto usando sympy (derivada de orden n)
        df_expr = sp.diff(f_expr, x, n)
        valor_exacto = float(df_expr.subs(x, x0))
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), tabla, h, nombre_regla, None
        
    except Exception as e:
        return None, None, None, None, None, None, f"Error en el cálculo de la derivada: {e}"
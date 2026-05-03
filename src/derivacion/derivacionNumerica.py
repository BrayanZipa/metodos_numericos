import sympy as sp
import math

"""
Calcula la derivada numérica de una función en un punto x0.
f_expr: función simbólica
x: símbolo de la variable (x)
x0: punto donde se evalúa la derivada
h: tamaño de paso
tipo_diferencia: 1 (Progresiva), 2 (Regresiva), 3 (Central)
"""
def derivadaDosPuntos(f_expr, x, x0, h, tipo_diferencia):
    try:
        if tipo_diferencia == 1:
            # Diferencia progresiva fórmula: D ≈ (f(x0+h) - f(x0)) / h
            fx0 = f_expr.subs(x, x0)
            fx1 = f_expr.subs(x, x0 + h)
            valor_aprox = (fx1 - fx0) / h
        elif tipo_diferencia == 2:
            # Diferencia regresiva fórmula: D ≈ (f(x0) - f(x0-h)) / h
            fx0 = f_expr.subs(x, x0)
            fx_1 = f_expr.subs(x, x0 - h)
            valor_aprox = (fx0 - fx_1) / h
        elif tipo_diferencia == 3:
            # Diferencia central fórmula: D ≈ (f(x0+h) - f(x0-h)) / (2h)
            fx1 = f_expr.subs(x, x0 + h)
            fx_1 = f_expr.subs(x, x0 - h)
            valor_aprox = (fx1 - fx_1) / (2 * h)
        else:
            return None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central)."
            
        # Calcular valor exacto usando sympy
        df_expr = sp.diff(f_expr, x)
        valor_exacto = df_expr.subs(x, x0)
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), None
        
    except Exception as e:
        return None, None, None, f"Error en el cálculo de la derivada: {e}"

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
        if tipo_diferencia == 1:
            nombre_regla = "Diferencia progresiva de 3 puntos"
            # Diferencia progresiva fórmula: D ≈ (-3f(x0) + 4f(x0+h) - f(x0+2h)) / (2h)
            valor_aprox = (-3 * float(f_expr.subs(x, x0)) + 4 * float(f_expr.subs(x, x0 + h)) - float(f_expr.subs(x, x0 + 2 * h))) / (2 * h)
        elif tipo_diferencia == 2:
            nombre_regla = "Diferencia regresiva de 3 puntos"
            # Diferencia regresiva fórmula: D ≈ (3f(x0) - 4f(x0-h) + f(x0-2h)) / (2h)
            valor_aprox = (3 * float(f_expr.subs(x, x0)) - 4 * float(f_expr.subs(x, x0 - h)) + float(f_expr.subs(x, x0 - 2 * h))) / (2 * h)
        elif tipo_diferencia == 3:
            nombre_regla = "Diferencia central de 3 puntos"
            # Diferencia central fórmula: D ≈ (f(x0+h) - f(x0-h)) / (2h)
            valor_aprox = (float(f_expr.subs(x, x0 + h)) - float(f_expr.subs(x, x0 - h))) / (2 * h)
        else:
            return None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central).", None
            
        # Calcular valor exacto usando sympy (primera derivada)
        df_expr = sp.diff(f_expr, x, 1)
        valor_exacto = float(df_expr.subs(x, x0))
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), None, nombre_regla
        
    except Exception as e:
        return None, None, None, f"Error en el cálculo de la derivada: {e}", None

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
        if tipo_diferencia == 1:
            nombre_regla = "Diferencia progresiva de 5 puntos"
            # Diferencia progresiva fórmula: D ≈ (-25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h)) / 12h
            v = -25 * float(f_expr.subs(x, x0)) + 48 * float(f_expr.subs(x, x0 + h)) - 36 * float(f_expr.subs(x, x0 + 2 * h)) + 16 * float(f_expr.subs(x, x0 + 3 * h)) - 3 * float(f_expr.subs(x, x0 + 4 * h))
            valor_aprox = v / (12 * h)
        elif tipo_diferencia == 2:
            nombre_regla = "Diferencia regresiva de 5 puntos"
            # Diferencia regresiva fórmula: D ≈ (25f(x) - 48f(x-h) + 36f(x-2h) - 16f(x-3h) + 3f(x-4h)) / 12h
            v = 25 * float(f_expr.subs(x, x0)) - 48 * float(f_expr.subs(x, x0 - h)) + 36 * float(f_expr.subs(x, x0 - 2 * h)) - 16 * float(f_expr.subs(x, x0 - 3 * h)) + 3 * float(f_expr.subs(x, x0 - 4 * h))
            valor_aprox = v / (12 * h)
        elif tipo_diferencia == 3:
            nombre_regla = "Diferencia central de 5 puntos"
            # Diferencia central fórmula: D ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / 12h
            v = float(f_expr.subs(x, x0 - 2 * h)) - 8 * float(f_expr.subs(x, x0 - h)) + 8 * float(f_expr.subs(x, x0 + h)) - float(f_expr.subs(x, x0 + 2 * h))
            valor_aprox = v / (12 * h)
        else:
            return None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central).", None
            
        # Calcular valor exacto usando sympy (primera derivada)
        df_expr = sp.diff(f_expr, x, 1)
        valor_exacto = float(df_expr.subs(x, x0))
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), None, nombre_regla
        
    except Exception as e:
        return None, None, None, f"Error en el cálculo de la derivada: {e}", None

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
            return None, None, None, "El orden de la derivada debe ser mayor o igual a 1."
            
        valor_aprox = 0
        if tipo_diferencia == 1:
            # Diferencia progresiva fórmula: D ≈ (1/h^n) * Sum[(-1)^(n-k) * comb(n,k) * f(x0+kh)]
            for k in range(n + 1):
                coef = ((-1)**(n - k)) * math.comb(n, k)
                valor_aprox += coef * float(f_expr.subs(x, x0 + k * h))
            valor_aprox = valor_aprox / (h**n)
            
        elif tipo_diferencia == 2:
            # Diferencia regresiva fórmula: D ≈ (1/h^n) * Sum[(-1)^k * comb(n,k) * f(x0-kh)]
            for k in range(n + 1):
                coef = ((-1)**k) * math.comb(n, k)
                valor_aprox += coef * float(f_expr.subs(x, x0 - k * h))
            valor_aprox = valor_aprox / (h**n)
            
        elif tipo_diferencia == 3:
            # Diferencia central fórmula: D ≈ (1/(2h)^n) * Sum[(-1)^k * comb(n,k) * f(x0 + (n - 2k)h)]
            for k in range(n + 1):
                coef = ((-1)**k) * math.comb(n, k)
                valor_aprox += coef * float(f_expr.subs(x, x0 + (n - 2 * k) * h))
            valor_aprox = valor_aprox / ((2 * h)**n)
            
        else:
            return None, None, None, "Tipo de diferencia no válido. Use 1 (Progresiva), 2 (Regresiva) o 3 (Central)."
            
        # Calcular valor exacto usando sympy (derivada de orden n)
        df_expr = sp.diff(f_expr, x, n)
        valor_exacto = df_expr.subs(x, x0)
        
        # Calcular error absoluto
        error_abs = abs(valor_exacto - valor_aprox)
        
        return float(valor_aprox), float(valor_exacto), float(error_abs), None
        
    except Exception as e:
        return None, None, None, f"Error en el cálculo de la derivada: {e}"
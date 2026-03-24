import sympy as sp
import math

"""
Calcula el polinomio de Taylor de grado n para f_expr centrado en x0.
f_expr: expresión de sympy que representa la función original
x: símbolo de la variable (x)
x0: punto de evaluación o centro alrededor del cual se aproxima
n: grado máximo del polinomio a calcular
x_eval: punto opcional en el que se desea evaluar el polinomio resultante
"""
def taylor(f_expr, x, x0, n, x_eval = None):
    polinomio = 0
    iteraciones = []
    
    try:
        for k in range(n + 1):
            # Calcular derivada k-ésima
            df_k = sp.diff(f_expr, x, k)
            # Evaluar derivada en x0
            df_k_x0 = df_k.subs(x, x0).evalf(6)
            
            # Calcular término de Taylor: f^(k)(x0) / k! * (x - x0)^k
            termino = (df_k_x0 / math.factorial(k)) * (x - x0)**k
            polinomio += termino
            
            # Calcular valor del término si hay x_eval
            val_termino = "N/A"
            val_acumulado = "N/A"
            if x_eval is not None:
                val_termino = float(termino.subs(x, x_eval).evalf())
                val_acumulado = float(polinomio.subs(x, x_eval).evalf())
                
            iteraciones.append({
                'k': k,
                'df_k': df_k,
                'df_k_x0': df_k_x0,
                'termino': str(sp.N(termino, 6)),
                'termino_simple': str(sp.expand(termino.evalf(6))),
                'val_termino': val_termino,
                'val_acumulado': val_acumulado
            })
            
        return sp.simplify(polinomio.evalf(6)), iteraciones, None
    except Exception as e:
        return None, None, f"Error al calcular por el polinomio de Taylor: {str(e)}"
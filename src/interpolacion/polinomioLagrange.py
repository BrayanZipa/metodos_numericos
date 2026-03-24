import sympy as sp

"""
Calcula el polinomio de Lagrange para un conjunto de puntos (x, y).
x_points: lista de valores x
y_points: lista de valores y
x: símbolo de la variable (x)
x_eval: punto opcional en el que se desea evaluar el polinomio resultante
"""
def lagrange(x_points, y_points, x, x_eval = None):
    n = len(x_points)
    polinomio = 0
    li_polinomios = []
    
    try:
        if len(x_points) != len(y_points):
            return None, None, "La cantidad de puntos x y f(x) deben ser iguales."
        
        for i in range(n):
            li = 1
            for j in range(n):
                if i != j:
                    li *= (x - x_points[j]) / (x_points[i] - x_points[j])
            
            li = sp.expand(li)
            termino = y_points[i] * li
            polinomio += termino
            
            # Calcular valor del término si hay x_eval
            val_li = "N/A"
            val_termino = "N/A"
            if x_eval is not None:
                val_li = float(li.subs(x, x_eval).evalf())
                val_termino = float(termino.subs(x, x_eval).evalf())
            
            li_polinomios.append({
                'i': i,
                'xi': x_points[i],
                'yi': y_points[i],
                'li': str(sp.expand(li.evalf(6))),
                'termino': str(sp.expand(termino.evalf(6))),
                'val_li': val_li,
                'val_termino': val_termino
            })
            
        polinomio = sp.simplify(polinomio.evalf(6))
        return polinomio, li_polinomios, None
    except Exception as e:
        return None, None, f"Error al calcular el polinomio de Lagrange: {str(e)}"

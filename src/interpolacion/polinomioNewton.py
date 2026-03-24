import sympy as sp

"""
Calcula el polinomio de Newton para un conjunto de puntos (x, y) usando diferencias divididas.
x_points: lista de valores x
y_points: lista de valores y
x: símbolo de la variable (x)
x_eval: punto opcional en el que se desea evaluar el polinomio resultante
"""
def newton(x_points, y_points, x, x_eval=None):
    n = len(x_points)
    
    try:
        if len(x_points) != len(y_points):
            return None, None, "La cantidad de puntos x y f(x) deben ser iguales."
        
        # Crear la tabla de diferencias divididas
        tabla = [[0.0] * n for _ in range(n)]
        
        # La primera columna son los valores de y
        for i in range(n):
            tabla[i][0] = float(y_points[i])
            
        # Calcular las diferencias divididas
        for j in range(1, n):
            for i in range(n - j):
                tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x_points[i+j] - x_points[i])

        # Construir el polinomio
        polinomio = tabla[0][0]
        termino_acumulado = 1
        
        pasos = []
        pasos.append({
            'i': 0,
            'xi': x_points[0],
            'fi': y_points[0],
            'coef': tabla[0][0],
            'termino': str(sp.N(sp.Float(tabla[0][0]), 6)),
            'termino_simple': str(sp.N(sp.Float(tabla[0][0]), 6)),
            'val_termino': float(tabla[0][0]) if x_eval is not None else "N/A"
        })
        
        for i in range(1, n):
            termino_acumulado *= (x - x_points[i-1])
            coeficiente = tabla[0][i]
            termino = coeficiente * termino_acumulado
            polinomio += termino
            
            # Calcular valor del término si hay x_eval
            val_termino = "N/A"
            if x_eval is not None:
                val_termino = float(termino.subs(x, x_eval).evalf())
            
            pasos.append({
                'i': i,
                'xi': x_points[i],
                'fi': y_points[i],
                'coef': coeficiente,
                'termino': str(sp.N(termino, 6)),
                'termino_simple': str(sp.expand(termino.evalf(6))),
                'val_termino': val_termino
            })
            
        polinomio = sp.simplify(polinomio.evalf(6))

        
        # Estructurar la tabla de diferencias divididas
        diff_table = []
        for i in range(n):
            fila = {'xi': x_points[i], 'fi': y_points[i]}
            for j in range(1, n - i):
                fila[f'diff_{j}'] = tabla[i][j]
            diff_table.append(fila)
            
        return polinomio, {'pasos': pasos, 'tabla': diff_table}, None
        
    except Exception as e:
        return None, None, f"Error al calcular el polinomio de Newton: {str(e)}"

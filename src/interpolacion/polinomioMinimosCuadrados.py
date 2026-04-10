import sympy as sp

"""
Calcula el polinomio de aproximación por el método de mínimos cuadrados.
x_points: lista de valores x
y_points: lista de valores y
x: símbolo de la variable (x)
grado: grado del polinomio a ajustar
x_eval: punto opcional en el que se desea evaluar el polinomio resultante
"""
def minimosCuadrados(x_points, y_points, x, grado, x_eval = None):
    try:
        n = len(x_points)
        if len(y_points) != n:
            return None, None, "La cantidad de puntos x y y deben ser iguales."
            
        # Construir matriz A (n x grado+1)
        A_list = []
        for i in range(n):
            fila = [float(x_points[i]**j) for j in range(grado + 1)]
            A_list.append(fila)
            
        A = sp.Matrix(A_list)
        Y = sp.Matrix([float(y) for y in y_points])
        
        # Resolver el sistema de ecuaciones: (A^T * A) * C = A^T * Y
        At = A.T
        AtA = At * A
        AtY = At * Y
        
        coeficientes = AtA.inv() * AtY
        coefs_list = [float(coeficientes[i]) for i in range(grado + 1)]
        
        # Construir polinomio
        polinomio = 0
        for i in range(grado + 1):
            polinomio += coefs_list[i] * (x**i)
            
        polinomio = sp.simplify(polinomio.evalf(6))
        
        # Calcular error cuadrático y r^2
        y_promedio = sum(y_points) / n
        st = sum((y - y_promedio)**2 for y in y_points)
        
        sr = 0
        for i in range(n):
            y_calc = float(polinomio.subs(x, x_points[i]))
            sr += (y_points[i] - y_calc)**2
            
        r2 = (st - sr) / st if st != 0 else 1
        
        val_eval = None
        if x_eval is not None:
            val_eval = float(polinomio.subs(x, x_eval).evalf())
            
        # Construir tabla de sumatorias
        tabla_sumatorias = []
        for i in range(n):
            fila = {}
            fila['x'] = float(x_points[i])
            fila['y'] = float(y_points[i])
            for j in range(2, 2 * grado + 1):
                fila[f'x^{j}'] = float(x_points[i]**j)
            for j in range(1, grado + 1):
                key = 'xy' if j == 1 else f'x^{j}y'
                fila[key] = float((x_points[i]**j) * y_points[i])
            tabla_sumatorias.append(fila)
            
        sumas = {}
        sumas['x'] = sum(row['x'] for row in tabla_sumatorias)
        sumas['y'] = sum(row['y'] for row in tabla_sumatorias)
        for j in range(2, 2 * grado + 1):
            sumas[f'x^{j}'] = sum(row[f'x^{j}'] for row in tabla_sumatorias)
        for j in range(1, grado + 1):
            key = 'xy' if j == 1 else f'x^{j}y'
            sumas[key] = sum(row[key] for row in tabla_sumatorias)

        reporte = {
            'A': A_list,
            'AtA': [[float(val) for val in fila] for fila in AtA.tolist()],
            'AtY': [float(val[0]) for val in AtY.tolist()],
            'coeficientes': coefs_list,
            'sr': sr,
            'st': st,
            'r2': r2,
            'val_eval': val_eval,
            'tabla_sumatorias': tabla_sumatorias,
            'sumatorias': sumas
        }
            
        return polinomio, reporte, None
        
    except sp.MatrixError:
        return None, None, "El sistema de ecuaciones es singular y no tiene solución única (A^T * A no es invertible)."
    except Exception as e:
        return None, None, f"Error al calcular mínimos cuadrados: {str(e)}"
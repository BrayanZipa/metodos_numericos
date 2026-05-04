import sympy as sp

"""
Calcula el polinomio de Newton para un conjunto de puntos (x, y) usando diferencias divididas.
x_points: lista de valores x
y_points: lista de valores y
x: símbolo de la variable (x)
tipo_diferencia: tipo de diferencia dividida a calcular (1: progresiva, 2: regresiva, 3: centrada)
x_eval: punto opcional en el que se desea evaluar el polinomio resultante
"""
def newton(x_points, y_points, x, tipo_diferencia, x_eval = None):
    n = len(x_points)
    
    try:
        if len(x_points) != len(y_points):
            return None, None, "La cantidad de puntos x y f(x) deben ser iguales."
        
        n = len(x_points)
        x_pts = x_points[:]
        y_pts = y_points[:]
        
        # Construir la tabla completa de diferencias divididas (siempre sobre el orden original)
        tabla_full = [[0.0] * n for _ in range(n)]
        for i in range(n):
            tabla_full[i][0] = float(y_pts[i])
        for j in range(1, n):
            for i in range(n - j):
                tabla_full[i][j] = (tabla_full[i+1][j-1] - tabla_full[i][j-1]) / (x_pts[i+j] - x_pts[i])

        # Selección del método y coeficientes
        coeficientes = []
        bases = [] # Listas de puntos (x - xi) para cada término

        if tipo_diferencia == 1: # Progresivas
            for j in range(n):
                coeficientes.append(tabla_full[0][j])
                bases.append(x_pts[:j])
        
        elif tipo_diferencia == 2: # Regresivas
            # Para regresivas usamos el último renglón de la tabla
            for j in range(n):
                coeficientes.append(tabla_full[n-j-1][j])
                bases.append(x_pts[n-j:])
        
        elif tipo_diferencia == 3: # Centradas (Lógica Stirling/Bessel)
            # Buscamos el centro
            if x_eval is not None:
                # Buscar índice más cercano al valor de evaluación
                idx_centro = min(range(n), key=lambda i: abs(x_pts[i] - x_eval))
            else:
                idx_centro = n // 2
            
            # Para centradas, el polinomio es el promedio de dos caminos (Gauss Forward y Backward)
            # Esto genera los promedios característicos de Stirling/Bessel
            def get_path_coefs(idx, order):
                # Genera los coeficientes siguiendo un camino zigzag
                path_pts = []
                # El orden del camino para Stirling/Gauss:
                # F: k, k+1, k-1, k+2, k-2 ...
                # B: k, k-1, k+1, k-2, k+2 ...
                curr_idx = idx
                path_pts.append(curr_idx)
                
                # Coeficiente d0
                coefs = [tabla_full[curr_idx][0]]
                bases_idx = [[]]
                
                for j in range(1, n):
                    # Elegir siguiente punto para el camino
                    # Esta es una simplificación: alternar arriba y abajo
                    if (j % 2 != 0 and order == 'F') or (j % 2 == 0 and order == 'B'):
                        next_idx = curr_idx - 1
                    else:
                        next_idx = curr_idx
                    
                    if next_idx < 0 or next_idx + j >= n:
                        # Si se sale de los bordes, intentar el otro lado
                        next_idx = curr_idx if next_idx < curr_idx else curr_idx - 1
                        if next_idx < 0 or next_idx + j >= n: break 
                    
                    coefs.append(tabla_full[next_idx][j])
                    bases_idx.append(path_pts[:]) # Bases son los puntos previos en el camino
                    path_pts.append(next_idx + j if next_idx < curr_idx else next_idx)
                    curr_idx = next_idx
                return coefs, [[x_pts[i] for i in base] for base in bases_idx]

            if n % 2 != 0: # Impar: Stirling centrado en un punto
                k = idx_centro
                c1, b1 = get_path_coefs(k, 'F')
                c2, b2 = get_path_coefs(k, 'B')
            else: # Par: Bessel centrado entre dos puntos
                k1 = idx_centro - (1 if idx_centro > 0 else 0)
                k2 = idx_centro if idx_centro > 0 else 1
                c1, b1 = get_path_coefs(k1, 'F')
                c2, b2 = get_path_coefs(k2, 'B')

            # Promediar ambos caminos para obtener los coeficientes finales
            # Esto implementa f(x-1,x0)+f(x0,x1) / 2 y similares
            max_len = min(len(c1), len(c2))
            for i in range(max_len):
                if b1[i] == b2[i]:
                    coeficientes.append((c1[i] + c2[i]) / 2)
                    bases.append(b1[i])
                else:
                    # Si las bases difieren, el promedio se hace sobre los términos completos
                    # Pero para los pasos, mostraremos el promedio de los coefs si las bases son similares
                    coeficientes.append((c1[i] + c2[i]) / 2)
                    bases.append(b1[i]) # Simplificación para la visualización
        
        else:
            return None, None, "Tipo de diferencia dividida no válida."

        # Construir el Polinomio y los Pasos
        polinomio = sp.Integer(0)
        pasos = []
        idx_centro_ref = idx_centro if 'idx_centro' in locals() else 0
        
        for i in range(len(coeficientes)):
            coef = coeficientes[i]
            base_pts = bases[i]
            
            termino_monomial = sp.Integer(1)
            for xp in base_pts:
                termino_monomial *= (x - xp)
            
            termino = sp.simplify(coef * termino_monomial)
            polinomio += termino
            
            val_term = "N/A"
            if x_eval is not None:
                try:
                    val_term = float(sp.sympify(termino).subs(x, x_eval).evalf())
                except:
                    val_term = "Error"

            pasos.append({
                'i': i,
                'xi': base_pts[-1] if base_pts else x_pts[idx_centro_ref] if tipo_diferencia == 3 else x_pts[i],
                'fi': y_pts[i] if i < n else "N/A",
                'coef': coef,
                'termino': str(sp.N(termino, 6)),
                'termino_simple': str(sp.expand(termino.evalf(6))),
                'val_termino': val_term
            })

        polinomio = sp.simplify(polinomio.evalf(6))

        diff_table = []
        for i in range(n):
            fila = {'xi': x_pts[i], 'fi': y_pts[i]}
            for j in range(1, n - i):
                fila[f'diff_{j}'] = tabla_full[i][j]
            diff_table.append(fila)

        return polinomio, {'pasos': pasos, 'tabla': diff_table}, None
        
    except Exception as e:
        return None, None, f"Error al calcular el polinomio de Newton: {str(e)}"



# def newton(x_points, y_points, x, tipo_diferencia, x_eval = None):
#     n = len(x_points)
    
#     try:
#         if len(x_points) != len(y_points):
#             return None, None, "La cantidad de puntos x y f(x) deben ser iguales."
        
#         if tipo_diferencia == 1:  # Diferencias Divididas Progresivas
#             x_pts = x_points[:]
#             y_pts = y_points[:]
#         elif tipo_diferencia == 2:  # Diferencias Divididas Regresivas
#             x_pts = x_points[::-1]
#             y_pts = y_points[::-1]
#         elif tipo_diferencia == 3:  # Diferencias Divididas Centradas
#             if x_eval is not None:
#                 centro = x_eval
#             else:
#                 centro = x_points[n // 2]
            
#             # puntos = list(zip(x_points, y_points))
#             # # Ordenar por cercanía al centro
#             # puntos.sort(key=lambda p: abs(p[0] - centro))
#             # x_pts = [p[0] for p in puntos]
#             # y_pts = [p[1] for p in puntos]



#             puntos = list(zip(x_points, y_points))
#             puntos.sort(key=lambda p: abs(p[0] - centro))

#             # Intercalar: x0, x+1, x-1, x+2, x-2, ...
#             puntos_ordenados = [puntos[0]]
#             izq = [p for p in puntos[1:] if p[0] < puntos[0][0]]
#             der = [p for p in puntos[1:] if p[0] > puntos[0][0]]

#             for i in range(max(len(izq), len(der))):
#                 if i < len(der):
#                     puntos_ordenados.append(der[i])
#                 if i < len(izq):
#                     puntos_ordenados.append(izq[i])

#             x_pts = [p[0] for p in puntos_ordenados]
#             y_pts = [p[1] for p in puntos_ordenados]




#         else:
#             return None, None, "Tipo de diferencia no válido."

#         # Crear la tabla de diferencias divididas
#         tabla = [[0.0] * n for _ in range(n)]
        
#         # La primera columna son los valores de y
#         for i in range(n):
#             tabla[i][0] = float(y_pts[i])
            
#         # Calcular las diferencias divididas
#         for j in range(1, n):
#             for i in range(n - j):
#                 tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x_pts[i+j] - x_pts[i])

#         # Construir el polinomio
#         polinomio = tabla[0][0]
#         termino_acumulado = 1
        
#         pasos = []
#         pasos.append({
#             'i': 0,
#             'xi': x_pts[0],
#             'fi': y_pts[0],
#             'coef': tabla[0][0],
#             'termino': str(sp.N(sp.Float(tabla[0][0]), 6)),
#             'termino_simple': str(sp.N(sp.Float(tabla[0][0]), 6)),
#             'val_termino': float(tabla[0][0]) if x_eval is not None else "N/A"
#         })
        
#         for i in range(1, n):
#             termino_acumulado *= (x - x_pts[i-1])
#             coeficiente = tabla[0][i]
#             termino = coeficiente * termino_acumulado
#             polinomio += termino
            
#             # Calcular valor del término si hay x_eval
#             val_termino = "N/A"
#             if x_eval is not None:
#                 val_termino = float(termino.subs(x, x_eval).evalf())
            
#             pasos.append({
#                 'i': i,
#                 'xi': x_pts[i],
#                 'fi': y_pts[i],
#                 'coef': coeficiente,
#                 'termino': str(sp.N(termino, 6)),
#                 'termino_simple': str(sp.expand(termino.evalf(6))),
#                 'val_termino': val_termino
#             })
            
#         polinomio = sp.simplify(polinomio.evalf(6))

#         # Estructurar la tabla de diferencias divididas
#         diff_table = []
#         for i in range(n):
#             fila = {'xi': x_pts[i], 'fi': y_pts[i]}
#             for j in range(1, n - i):
#                 fila[f'diff_{j}'] = tabla[i][j]
#             diff_table.append(fila)
            
#         return polinomio, {'pasos': pasos, 'tabla': diff_table}, None
        
#     except Exception as e:
#         return None, None, f"Error al calcular el polinomio de Newton: {str(e)}"

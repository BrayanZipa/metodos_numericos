import sympy as sp
from tabulate import tabulate
from metodos.metodoBiseccion import biseccion
from metodos.metodoNewton import newtonRaphson
from metodos.metodoSecante import secante
from metodos.metodoPuntoFijo import puntoFijo
from interpolacion.polinomioTaylor import taylor
from interpolacion.polinomioLagrange import lagrange
from interpolacion.polinomioNewton import newton
from interpolacion.polinomioMinimosCuadrados import minimosCuadrados
from derivacion.derivacionNumerica import derivadaDosPuntos, derivadaTresPuntos, derivadaCincoPuntos, derivadaOrdenSuperior
from integracion.reglaTrapecio import trapecioSimple, trapecioCompuesta
from integracion.reglaSimpson import simpsonUnTercioSimple, simpsonUnTercioCompuesta, simpsonTresOctavosSimple, simpsonTresOctavosCompuesta
from integracion.reglaBoole import booleSimple, booleCompuesta
from ecuacionDiferencial.metodoEuler import euler
from graficar import graficarMetodos, graficarInterpolacion, graficarIntegracion, graficarDerivacion, graficarEcuacionDiferencial
from generarArchivo import crearGgb

def menuMetodos():
    x = sp.symbols('x')
    
    while True:
        print("\n--- MÉTODOS NUMÉRICOS POR BRAYAN ZIPA ---")
        print("\nSintaxis recomendada: sqrt(x), log(x), exp(x), sin(x), cos(x), x**n")
        print("\n--- Seleccione el método a utilizar ---\n")
        print("1. Método de Bisección")
        print("2. Método de Newton-Raphson")
        print("3. Método de la Secante")
        print("4. Método de Punto Fijo")
        print("5. Volver al menú principal")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '5':
            break
            
        if opcion not in ['1', '2', '3', '4']:
            print("Opción no válida.")
            continue

        try:
            if opcion == '1':
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: x**3 - 25:  ")
            elif opcion == '2':
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: x**3 -2*x - 5:  ")
            elif opcion == '3':
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: cos(x) - x:  ")
            elif opcion == '4':
                expr_str = input("Ingrese la función g(x) tal que x = g(x) ----> ejemplo de sintaxis: sqrt((10 - x**3)/4):  ")
            
            f_expr = sp.parse_expr(expr_str)
            tol = float(input("Ingrese el máximo error permitido (ej: 0.001): "))
            
            if opcion == '1':
                a = float(input("Ingrese el límite inferior (a): "))
                b = float(input("Ingrese el límite superior (b): "))
                
                iteraciones, error_msg = biseccion(f_expr, x, a, b, tol)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Método de Bisección:")
                    headers = ["Iter", "a", "b", "pn (a+b)/2", "f(a)", "f(pn)", "f(a)*f(pn)", "Error"]
                    data = [[i['iter'], f"{i['a']:.6f}", f"{i['b']:.6f}", f"{i['pn']:.6f}", f"{i['f(a)']:.6f}", f"{i['f(pn)']:.6f}", f"{i['f(a)*f(pn)']:.6f}", f"{i['error']:.6f}"] for i in iteraciones]
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    raiz_final = iteraciones[-1]['pn']
                    graficarMetodos(f_expr, x, raiz_final, a, b)

            elif opcion == '2':
                x0 = float(input("Ingrese el valor inicial (x0): "))
                
                iteraciones, error_msg = newtonRaphson(f_expr, x, x0, tol)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Método de Newton-Raphson:")
                    headers = ["Iter", "xi", "f(xi)", "df(xi)", "xi+1", "Error"]
                    data = [[i['iter'], f"{i['xi']:.6f}", f"{i['f(xi)']:.6f}", f"{i['df(xi)']:.6f}", f"{i['xi+1']:.6f}", f"{i['error']:.6f}"] for i in iteraciones]
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    raiz_final = iteraciones[-1]['xi+1']
                    graficarMetodos(f_expr, x, raiz_final)
            
            elif opcion == '3':
                x0 = float(input("Ingrese el valor inicial (x0): "))
                x1 = float(input("Ingrese el valor inicial (x1): "))
                
                iteraciones, error_msg = secante(f_expr, x, x0, x1, tol)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Método de la Secante:")
                    headers = ["Iter", "x0", "x1", "f(x0)", "f(x1)", "xi+1", "Error"]
                    data = [[i['iter'], f"{i['x0']:.6f}", f"{i['x1']:.6f}", f"{i['f(x0)']:.6f}", f"{i['f(x1)']:.6f}", f"{i['xi+1']:.6f}", f"{i['error']:.6f}"] for i in iteraciones]
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    raiz_final = iteraciones[-1]['xi+1']
                    graficarMetodos(f_expr, x, raiz_final)
            
            elif opcion == '4':
                x0 = float(input("Ingrese el valor inicial (x0): "))
                
                iteraciones, error_msg = puntoFijo(f_expr, x, x0, tol)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Método de Punto Fijo:")
                    headers = ["Iter", "xi", "xi+1 (g(xi))", "Error"]
                    data = [[i['iter'], f"{i['xi']:.6f}", f"{i['g(xi)']:.6f}", f"{i['error']:.6f}"] for i in iteraciones]
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    raiz_final = iteraciones[-1]['g(xi)']
                    graficarMetodos(f_expr, x, raiz_final)

        except Exception as e:
            print(f"Error al procesar los datos: {e}")

def menuInterpolacion():
    x = sp.symbols('x')
    
    while True:
        print("\n--- MÉTODOS DE INTERPOLACIÓN POR BRAYAN ZIPA ---")
        print("\nSintaxis recomendada: sqrt(x), log(x), exp(x), sin(x), cos(x), x**n")
        print("\n--- Seleccione el método a utilizar ---\n")
        print("1. Polinomio de Taylor")
        print("2. Polinomio de Lagrange")
        print("3. Polinomio de Newton")
        print("4. Polinomio por el método de mínimos cuadrados")
        print("5. Volver al menú principal")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '5':
            break
            
        if opcion not in ['1', '2', '3', '4', '5']:
            print("Opción no válida.")
            continue
            
        try:
            if opcion == '1':
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: exp(x), sin(x):  ")
                f_expr = sp.parse_expr(expr_str)
                x0 = float(input("Ingrese el punto de evaluación x0: "))
                n = int(input("Ingrese el grado del polinomio (n): "))
                
                x_eval_str = input("Ingrese el valor x a evaluar (deje en blanco para solo ver el polinomio): ")
                x_eval = float(x_eval_str) if x_eval_str.strip() else None
                
                polinomio, iteraciones, error_msg = taylor(f_expr, x, x0, n, x_eval)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Polinomio de Taylor:")
                    headers = ["n", "f^n(x)", "f^n(x0)", "Término f^n(x0)/n! * (x-x0)^n", "Término f^n(x0)/n! * (x-x0)^n simplificado"]
                    if x_eval is not None:
                        headers.extend([f"Valor Término en x={x_eval}", "Suma Acumulada"])
                        
                    data = []
                    for i in iteraciones:
                        row = [i['k'], i['df_k'], f"{float(i['df_k_x0']):.6g}", i['termino'], i['termino_simple']]
                        if x_eval is not None:
                            row.extend([f"{float(i['val_termino']):.6g}", f"{float(i['val_acumulado']):.6g}"])
                        data.append(row)
                        
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    print(f"\nPolinomio de Taylor de grado {n}:")
                    print(f"P(x) = {polinomio}")
                    
                    if x_eval is not None:
                        valor_real = float(f_expr.subs(x, x_eval))
                        valor_aprox = float(polinomio.subs(x, x_eval))
                        error_abs = abs(valor_real - valor_aprox)
                        
                        print(f"\nValor Real f({x_eval}) = {valor_real:.6g}")
                        print(f"Valor Aproximado P({x_eval}) = {valor_aprox:.6g}")
                        print(f"Error Absoluto = {error_abs:.6g}")
                        
                        if valor_real != 0:
                            error_rel_unitario = error_abs / abs(valor_real)
                            error_rel_porcentual = error_rel_unitario * 100
                            print(f"Error Relativo = {error_rel_unitario:.6g}")
                            print(f"Error Relativo (%) = {error_rel_porcentual:.6g}%")
                        else:
                            print("Error Relativo = N/A")
                            print("Error Relativo (%) = N/A (división por cero, valor real es 0)")
                    
                    a_graf, b_graf = x0 - 2, x0 + 2
                    if x_eval is not None:
                        margen = abs(x_eval - x0)
                        if margen == 0:
                            margen = 2
                        a_graf = min(x0, x_eval) - margen * 0.5
                        b_graf = max(x0, x_eval) + margen * 0.5

                    puntos_ggb = [(x0, float(f_expr.subs(x, x0)))]
                    if x_eval is not None:
                        puntos_ggb.append((x_eval, float(polinomio.subs(x, x_eval))))
                    
                    crearGgb([f_expr, polinomio], puntos_ggb, f"pol_taylor_n_{n}_x0_{x0}")
                    graficarInterpolacion(f_expr, polinomio, x, x0=x0, a=a_graf, b=b_graf)
            
            elif opcion == '2':
                num_puntos = int(input("Ingrese la cantidad de puntos (n): "))
                puntos_x = []
                puntos_y = []
                for i in range(num_puntos):
                    px = float(input(f"x[{i}]: "))
                    py = float(input(f"y[{i}]: "))
                    puntos_x.append(px)
                    puntos_y.append(py)
                
                x_eval_str = input("Ingrese el valor x a evaluar (deje en blanco para solo ver el polinomio): ")
                x_eval = float(x_eval_str) if x_eval_str.strip() else None
                
                polinomio, li_polinomios, error_msg = lagrange(puntos_x, puntos_y, x, x_eval)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Polinomio de Lagrange:")
                    headers = ["i", "xi", "f(xi)", "Li(x)", "Li(xi)*f(xi)"]
                    if x_eval is not None:
                        headers.extend([f"Li({x_eval})", f"Término en x={x_eval}"])
                    
                    data = []
                    for i in li_polinomios:
                        row = [i['i'], i['xi'], i['yi'], i['li'], i['termino']]
                        if x_eval is not None:
                            row.extend([f"{i['val_li']:.6g}", f"{i['val_termino']:.6g}"])
                        data.append(row)
                    
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    print(f"\nPolinomio de Lagrange P(x):")
                    print(f"P(x) = {polinomio}")
                    
                    if x_eval is not None:
                        valor_aprox = float(polinomio.subs(x, x_eval))
                        print(f"\nValor Aproximado P({x_eval}) = {valor_aprox:.6g}")
                    
                    puntos_ggb = list(zip(puntos_x, puntos_y))
                    if x_eval is not None:
                        puntos_ggb.append((x_eval, float(polinomio.subs(x, x_eval))))

                    crearGgb([polinomio], puntos_ggb, f"pol_lagrange_n_{num_puntos}")
                    graficarInterpolacion(None, polinomio, x, puntos_x=puntos_x, puntos_y=puntos_y, metodo="Lagrange")

            elif opcion == '3':
                num_puntos = int(input("Ingrese la cantidad de puntos (n): "))
                puntos_x = []
                puntos_y = []
                for i in range(num_puntos):
                    px = float(input(f"x[{i}]: "))
                    py = float(input(f"y[{i}]: "))
                    puntos_x.append(px)
                    puntos_y.append(py)

                print("\n1. Diferencias Divididas Progresivas")
                print("2. Diferencias Divididas Regresivas")
                print("3. Diferencias Divididas Centradas")
                tipo_diferencia = int(input("\nElija que tipo de diferencia dividida a calcular: "))

                if tipo_diferencia not in [1, 2, 3]:
                    print("Opción no válida.")
                    continue
                
                x_eval_str = input("Ingrese el valor x a evaluar (deje en blanco para solo ver el polinomio): ")
                x_eval = float(x_eval_str) if x_eval_str.strip() else None
                
                polinomio, data, error_msg = newton(puntos_x, puntos_y, x, tipo_diferencia, x_eval)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Polinomio de Newton:")
                    
                    # Tabla de Diferencias Divididas
                    tabla_diff = data['tabla']
                    
                    # Ajustar headers según las diferencias presentes
                    max_diffs = 0
                    if tabla_diff:
                        max_diffs = max(len([k for k in r.keys() if k.startswith('diff_')]) for r in tabla_diff)
                    
                    headers_tabla = ["i", "xi", "f[xi]"]
                    for j in range(1, max_diffs + 1):
                        headers_tabla.append(f"Diff {j}")

                    filas_tabla = []
                    for i, row in enumerate(tabla_diff):
                        fila = [i, row['xi'], row['fi']]
                        for j in range(1, max_diffs + 1):
                            val = row.get(f'diff_{j}', "")
                            fila.append(f"{val:.6f}" if isinstance(val, (int, float)) else val)
                        filas_tabla.append(fila)
                        
                    print("\nTabla de Diferencias Divididas:")
                    print(tabulate(filas_tabla, headers=headers_tabla, tablefmt="grid"))
                    
                    # Términos del Polinomio
                    print("\nPasos para construir el Polinomio:")
                    headers_pasos = ["i", "Coeficiente", "Término ai * (x-x0)...", "Término ai * (x-x0)... simplificado"]
                    if x_eval is not None:
                        headers_pasos.append(f"Valor en x={x_eval}")
                    
                    filas_pasos = []
                    for p in data['pasos']:
                        fila = [p['i'], f"{p['coef']:.6f}", p['termino'], p['termino_simple']]
                        if x_eval is not None:
                            fila.append(f"{p['val_termino']:.6g}")
                        filas_pasos.append(fila)
                    
                    print(tabulate(filas_pasos, headers=headers_pasos, tablefmt="grid"))
                    
                    print(f"\nPolinomio de Newton P(x):")
                    print(f"P(x) = {polinomio}")
                    
                    if x_eval is not None:
                        valor_aprox = float(polinomio.subs(x, x_eval))
                        print(f"\nValor Aproximado P({x_eval}) = {valor_aprox:.6g}")
                    
                    puntos_ggb = list(zip(puntos_x, puntos_y))
                    if x_eval is not None:
                        puntos_ggb.append((x_eval, float(polinomio.subs(x, x_eval))))

                    crearGgb([polinomio], puntos_ggb, f"pol_newton_n_{num_puntos}")
                    graficarInterpolacion(None, polinomio, x, puntos_x=puntos_x, puntos_y=puntos_y, metodo="Newton")

            elif opcion == '4':
                num_puntos = int(input("Ingrese la cantidad de puntos (n): "))
                puntos_x = []
                puntos_y = []
                for i in range(num_puntos):
                    px = float(input(f"x[{i}]: "))
                    py = float(input(f"y[{i}]: "))
                    puntos_x.append(px)
                    puntos_y.append(py)
                
                grado = int(input("Ingrese el grado del polinomio a ajustar (m): "))
                
                x_eval_str = input("Ingrese el valor x a evaluar (deje en blanco para solo ver el polinomio): ")
                x_eval = float(x_eval_str) if x_eval_str.strip() else None
                
                polinomio, reporte, error_msg = minimosCuadrados(puntos_x, puntos_y, x, grado, x_eval)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Mínimos Cuadrados:")
                    
                    print("\nTabla de Sumatorias:")
                    tabla_sum = reporte['tabla_sumatorias']
                    sumas = reporte['sumatorias']
                    
                    if tabla_sum:
                        headers_sum = ["i"] + list(tabla_sum[0].keys())
                        filas_sum = []
                        for i_row, row in enumerate(tabla_sum):
                            fila = [i_row] + [f"{row[k]:.6f}" if isinstance(row[k], (int, float)) else row[k] for k in headers_sum[1:]]
                            filas_sum.append(fila)
                        
                        fila_sumatoria = ["Σ"] + [f"{sumas[k]:.6f}" for k in headers_sum[1:]]
                        filas_sum.append(fila_sumatoria)
                        
                        print(tabulate(filas_sum, headers=headers_sum, tablefmt="grid"))

                    print("\nSistema Mínimos Cuadrados Normales (A^T * A * a = A^T * y):")
                    m_ata = reporte['AtA']
                    m_aty = reporte['AtY']
                    
                    data_sistema = []
                    headers_sistema = [f"a{i}" for i in range(grado + 1)] + ["=", "b"]
                    for i in range(grado + 1):
                        fila = [f"{val:.6f}" for val in m_ata[i]] + ["="] + [f"{m_aty[i]:.6f}"]
                        data_sistema.append(fila)
                        
                    print(tabulate(data_sistema, headers=headers_sistema, tablefmt="grid"))
                    
                    print("\nCoeficientes obtenidos:")
                    data_coef = []
                    for i, coef in enumerate(reporte['coeficientes']):
                        data_coef.append([f"a{i}", f"{coef:.6f}"])
                    print(tabulate(data_coef, headers=["Coeficiente", "Valor"], tablefmt="grid"))
                    
                    # print("\nMétricas de error:")
                    # print(f"Error sumatoria de cuadrados residuales (Sr): {reporte['sr']:.6f}")
                    # print(f"Sumatoria total (St): {reporte['st']:.6f}")
                    # print(f"Coeficiente de determinación r^2: {reporte['r2']:.6f}")

                    print(f"\nPolinomio por mínimos cuadrados de grado {grado}:")
                    print(f"P(x) = {polinomio}")
                    
                    if x_eval is not None:
                        valor_aprox = float(polinomio.subs(x, x_eval))
                        print(f"\nValor Aproximado P({x_eval}) = {valor_aprox:.6f}")
                    
                    puntos_ggb = list(zip(puntos_x, puntos_y))
                    if x_eval is not None:
                        puntos_ggb.append((x_eval, float(polinomio.subs(x, x_eval))))

                    crearGgb([polinomio], puntos_ggb, f"pol_minimos_cuadrados_m_{grado}")
                    graficarInterpolacion(None, polinomio, x, puntos_x=puntos_x, puntos_y=puntos_y, metodo="Mínimos Cuadrados")
                
        except Exception as e:
            print(f"Error al procesar los datos: {e}")

def menuDerivacionNumerica():
    x = sp.symbols('x')
    
    while True:
        print("\n--- MÉTODOS DE DERIVACIÓN NUMÉRICA POR BRAYAN ZIPA ---")
        print("\nSintaxis recomendada: sqrt(x), log(x), exp(x), sin(x), cos(x), x**n")
        print("\n--- Seleccione el método a utilizar ---\n")
        print("1. Fórmula de 2 puntos")
        print("2. Fórmula de 3 puntos (Mayor precisión O(h^2))")
        print("3. Fórmula de 5 puntos (Alta precisión O(h^4))")
        print("4. Fórmula para derivada de orden superior (2da, 3ra...)")
        print("5. Volver al menú principal")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '5':
            break

        if opcion not in ['1', '2', '3', '4']:
            print("Opción no válida.")
            continue
            
        try:
            expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: exp(x), sin(x):  ")
            x0 = float(input("Ingrese el punto donde desea derivar (x0): "))
            h = float(input("Ingrese el tamaño de paso (h): "))
            
            print("\nTipos de diferencia:")
            print("1. Diferencia progresiva")
            print("2. Diferencia regresiva")
            print("3. Diferencia central")
            tipo_diferencia = int(input("\nElija qué tipo de diferencia usar: "))
            
            if expr_str.strip():
                f_expr = sp.parse_expr(expr_str)
            else:
                # Ingresar puntos manualmente y construir polinomio por interpolación
                num_puntos = int(input("Ingrese la cantidad de puntos: "))
                puntos_x_input = []
                puntos_y_input = []
                for i in range(num_puntos):
                    px = float(input(f"x[{i}]: "))
                    py = float(input(f"y[{i}]: "))
                    puntos_x_input.append(px)
                    puntos_y_input.append(py)
                
                # Intentar construir polinomio con Lagrange primero, si falla usar Newton
                f_expr = None
                try:
                    polinomio_lag, _, error_lag = lagrange(puntos_x_input, puntos_y_input, x)
                    if error_lag is None and polinomio_lag is not None:
                        f_expr = polinomio_lag
                        print(f"\nPolinomio construido por Lagrange: f(x) = {f_expr}")
                except Exception:
                    pass
                
                if f_expr is None:
                    try:
                        polinomio_new, _, error_new = newton(puntos_x_input, puntos_y_input, x, tipo_diferencia)
                        if error_new is None and polinomio_new is not None:
                            f_expr = polinomio_new
                            print(f"\nPolinomio construido por Newton: f(x) = {f_expr}")
                    except Exception:
                        pass
                
                if f_expr is None:
                    print("Error: No se pudo construir un polinomio con los puntos ingresados.")
                    continue
            
            tabla = None
            error_teorico = None
            error_msg = None
            nombre_regla = None
            orden = 1
            
            if opcion == '1':
                valor_aprox, valor_exacto, error_abs, error_teorico, tabla, h, nombre_regla, error_msg = derivadaDosPuntos(f_expr, x, x0, h, tipo_diferencia)
                metodo_nombre = "2 Puntos"
            elif opcion == '2':
                valor_aprox, valor_exacto, error_abs, error_teorico, tabla, h, nombre_regla, error_msg = derivadaTresPuntos(f_expr, x, x0, h, tipo_diferencia)
                metodo_nombre = "3 Puntos"
            elif opcion == '3':
                valor_aprox, valor_exacto, error_abs, error_teorico, tabla, h, nombre_regla, error_msg = derivadaCincoPuntos(f_expr, x, x0, h, tipo_diferencia)
                metodo_nombre = "5 Puntos"
            elif opcion == '4':
                orden = int(input("Ingrese el orden de la derivada (2 para segunda, 3 para tercera...): "))
                valor_aprox, valor_exacto, error_abs, tabla, h, nombre_regla, error_msg = derivadaOrdenSuperior(f_expr, x, x0, h, orden, tipo_diferencia)
                metodo_nombre = f"Orden Superior ({orden})"
            
            if error_msg:
                print(error_msg)
            else:
                # Mostrar tabla de puntos evaluados
                print("\nTabla de puntos evaluados:")
                headers = ["i", "xi", "f(xi)"]
                data = [[row['i'], f"{row['x']:.6f}", f"{row['f(x)']:.6f}"] for row in tabla]
                print(tabulate(data, headers=headers, tablefmt="grid"))
                
                # Resumen de resultados
                print(f"\nResultados Derivada Numérica ({nombre_regla}):")
                
                derivada_simbolica = sp.diff(f_expr, x, orden)
                print(f"\nFunción original: f(x) = {f_expr}")
                print(f"Derivada real de la función original (orden {orden}): f{''.join([chr(39)]*orden)}(x) = {derivada_simbolica}")
                
                print(f"\nPunto de evaluación (x0): {x0}")
                print(f"Tamaño de paso (h): {h:.6g}")
                print(f"\nValor Aproximado: {valor_aprox:.6f}")
                print(f"Valor Exacto: {valor_exacto:.6f}")
                print(f"Error Absoluto: {error_abs:.6g}")
                
                if error_teorico is not None:
                    print(f"Error Teórico: {error_teorico:.6g}")
                
                # Generar gráfica y archivo GGB
                puntos_x = [row['x'] for row in tabla]
                puntos_y = [row['f(x)'] for row in tabla]
                
                puntos_ggb = list(zip(puntos_x, puntos_y))
                nombre_archivo = f"derivada_{metodo_nombre.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
                crearGgb([f_expr], puntos_ggb, nombre_archivo)
                
                graficarDerivacion(f_expr, x, x0, puntos_x, puntos_y, metodo=nombre_regla)
                
        except Exception as e:
            print(f"Error al procesar los datos: {e}")

def menuIntegracionNumerica():
    x = sp.symbols('x')

    while True:
        print("\n--- MÉTODOS DE INTEGRACIÓN NUMÉRICA POR BRAYAN ZIPA ---")
        print("\nSintaxis recomendada: sqrt(x), log(x), exp(x), sin(x), cos(x), x**n")
        print("\n--- Seleccione el método a utilizar ---\n")
        print("1. Regla del Trapecio")
        print("2. Regla de Simpson")
        print("3. Regla de Boole")
        print("4. Volver al menú principal")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '4':
            break
            
        if opcion not in ['1', '2', '3']:
            print("Opción no válida.")
            continue

        try:
            if opcion == '1':
                print("\n--- Regla del Trapecio ---")
                print("1. Trapecio Simple")
                print("2. Trapecio Compuesta")
                sub_opcion = input("\nElija una opción: ")
                
                if sub_opcion not in ['1', '2']:
                    print("Opción no válida.")
                    continue
                
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: exp(x), sin(x):  ")
                if expr_str.strip():
                    f_expr = sp.parse_expr(expr_str)
                else:
                    num_puntos = int(input("Ingrese la cantidad de puntos: "))
                    puntos_x_input = []
                    puntos_y_input = []
                    for i in range(num_puntos):
                        px = float(input(f"x[{i}]: "))
                        py = float(input(f"y[{i}]: "))
                        puntos_x_input.append(px)
                        puntos_y_input.append(py)
                    
                    f_expr = None
                    try:
                        polinomio_lag, _, error_lag = lagrange(puntos_x_input, puntos_y_input, x)
                        if error_lag is None and polinomio_lag is not None:
                            f_expr = polinomio_lag
                            print(f"\nPolinomio construido por Lagrange: f(x) = {f_expr}")
                    except Exception:
                        pass
                    
                    if f_expr is None:
                        try:
                            polinomio_new, _, error_new = newton(puntos_x_input, puntos_y_input, x, 1)
                            if error_new is None and polinomio_new is not None:
                                f_expr = polinomio_new
                                print(f"\nPolinomio construido por Newton: f(x) = {f_expr}")
                        except Exception:
                            pass
                    
                    if f_expr is None:
                        print("Error: No se pudo construir un polinomio con los puntos ingresados.")
                        continue

                a = float(input("Ingrese el límite inferior (a): "))
                b = float(input("Ingrese el límite superior (b): "))
                
                if sub_opcion == '1':
                    aprox, exacto, error, tabla, h, error_msg = trapecioSimple(f_expr, x, a, b)
                    metodo_nombre = "Trapecio Simple"
                elif sub_opcion == '2':
                    n = int(input("Ingrese el número de subintervalos (n): "))
                    aprox, exacto, error, tabla, h, error_msg = trapecioCompuesta(f_expr, x, a, b, n)
                    metodo_nombre = "Trapecio Compuesta"
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nTabla de puntos evaluados:")
                    headers = ["i", "xi", "f(xi)"]
                    data = [[row['i'], f"{row['x']:.6f}", f"{row['f(x)']:.6f}"] for row in tabla]
                    
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    f0 = tabla[0]['f(x)']
                    fn = tabla[-1]['f(x)']
                    
                    suma_interior = sum(row['f(x)'] for row in tabla[1:-1])
                    print("\nDesglose del cálculo:")
                    print(f"Σf(xi) internos = {suma_interior:.6f}")
                    print(f"2 * Σf(xi) internos = {2 * suma_interior:.6f}")
                    
                    total_suma = f0 + 2 * suma_interior + fn
                    print(f"\nSuma total: f(x0) + 2*Σf(xi) + f(xn) = {total_suma:.6f}")
                    
                    print(f"\nResultados Regla del Trapecio ({metodo_nombre}):")
                    
                    integral_simbolica = sp.integrate(f_expr, x)
                    print(f"\nFunción original: f(x) = {f_expr}")
                    print(f"Integral real de la función original: F(x) = {integral_simbolica}")
                    
                    print(f"\nLímites de integración: [{a}, {b}]")
                    print(f"Tamaño de paso (h): {h:.6g}")
                    print(f"\nValor Aproximado: {aprox:.6f}")
                    print(f"Valor Exacto: {exacto:.6f}")
                    print(f"Error Absoluto: {error:.6g}")
                    
                    puntos_x = [row['x'] for row in tabla]
                    puntos_y = [row['f(x)'] for row in tabla]
                    
                    puntos_ggb = list(zip(puntos_x, puntos_y))
                    nombre_archivo = "integral_trapecio_simple" if sub_opcion == '1' else f"integral_trapecio_compuesta_n_{n}"
                    crearGgb([f_expr], puntos_ggb, nombre_archivo)
                    
                    graficarIntegracion(f_expr, x, a, b, puntos_x, puntos_y, metodo=metodo_nombre)

            elif opcion == '2':
                print("\n--- Regla de Simpson ---")
                print("1. Simpson 1/3 Simple")
                print("2. Simpson 1/3 Compuesta")
                print("3. Simpson 3/8 Simple")
                print("4. Simpson 3/8 Compuesta")
                sub_opcion = input("\nElija una opción: ")
                
                if sub_opcion not in ['1', '2', '3', '4']:
                    print("Opción no válida.")
                    continue
                
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: exp(x), sin(x):  ")
                if expr_str.strip():
                    f_expr = sp.parse_expr(expr_str)
                else:
                    num_puntos = int(input("Ingrese la cantidad de puntos: "))
                    puntos_x_input = []
                    puntos_y_input = []
                    for i in range(num_puntos):
                        px = float(input(f"x[{i}]: "))
                        py = float(input(f"y[{i}]: "))
                        puntos_x_input.append(px)
                        puntos_y_input.append(py)
                    
                    f_expr = None
                    try:
                        polinomio_lag, _, error_lag = lagrange(puntos_x_input, puntos_y_input, x)
                        if error_lag is None and polinomio_lag is not None:
                            f_expr = polinomio_lag
                            print(f"\nPolinomio construido por Lagrange: f(x) = {f_expr}")
                    except Exception:
                        pass
                    
                    if f_expr is None:
                        try:
                            polinomio_new, _, error_new = newton(puntos_x_input, puntos_y_input, x, 1)
                            if error_new is None and polinomio_new is not None:
                                f_expr = polinomio_new
                                print(f"\nPolinomio construido por Newton: f(x) = {f_expr}")
                        except Exception:
                            pass
                    
                    if f_expr is None:
                        print("Error: No se pudo construir un polinomio con los puntos ingresados.")
                        continue
                
                a = float(input("Ingrese el límite inferior (a): "))
                b = float(input("Ingrese el límite superior (b): "))
                
                if sub_opcion == '1':
                    aprox, exacto, error, tabla, h, error_msg = simpsonUnTercioSimple(f_expr, x, a, b)
                    metodo_nombre = "Simpson 1/3 Simple"
                elif sub_opcion == '2':
                    n = int(input("Ingrese el número de subintervalos (n) [debe ser par]: "))
                    aprox, exacto, error, tabla, h, error_msg = simpsonUnTercioCompuesta(f_expr, x, a, b, n)
                    metodo_nombre = "Simpson 1/3 Compuesta"
                elif sub_opcion == '3':
                    aprox, exacto, error, tabla, h, error_msg = simpsonTresOctavosSimple(f_expr, x, a, b)
                    metodo_nombre = "Simpson 3/8 Simple"
                elif sub_opcion == '4':
                    n = int(input("Ingrese el número de subintervalos (n) [debe ser múltiplo de 3]: "))
                    aprox, exacto, error, tabla, h, error_msg = simpsonTresOctavosCompuesta(f_expr, x, a, b, n)
                    metodo_nombre = "Simpson 3/8 Compuesta"
                    
                if error_msg:
                    print(error_msg)
                else:
                    print("\nTabla de puntos evaluados:")
                    headers = ["i", "xi", "f(xi)"]
                    data = [[row['i'], f"{row['x']:.6f}", f"{row['f(x)']:.6f}"] for row in tabla]
                    
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    f0 = tabla[0]['f(x)']
                    fn = tabla[-1]['f(x)']
                    
                    if sub_opcion in ['1', '2']:
                        suma_impares = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 2 != 0)
                        suma_pares = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 2 == 0)
                        print("\nDesglose del cálculo:")
                        print(f"Σf(x_pares) = {suma_pares:.6f}")
                        print(f"2 * Σf(x_pares) = {2 * suma_pares:.6f}")
                        print(f"Σf(x_impares) = {suma_impares:.6f}")
                        print(f"4 * Σf(x_impares) = {4 * suma_impares:.6f}")
                        
                        total_suma = f0 + 4 * suma_impares + 2 * suma_pares + fn
                        print(f"\nSuma total: f(x0) + 4*Σf(x_impares) + 2*Σf(x_pares) + f(xn) = {total_suma:.6f}")
                        
                    elif sub_opcion in ['3', '4']:
                        suma_resto = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 3 != 0)
                        suma_multiplos_3 = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 3 == 0)
                        print("\nDesglose del cálculo:")
                        print(f"Σf(múltiplos de 3) = {suma_multiplos_3:.6f}")
                        print(f"2 * Σf(múltiplos de 3) = {2 * suma_multiplos_3:.6f}")
                        print(f"Σf(resto) = {suma_resto:.6f}")
                        print(f"3 * Σf(resto) = {3 * suma_resto:.6f}")
                        
                        total_suma = f0 + 3 * suma_resto + 2 * suma_multiplos_3 + fn
                        print(f"\nSuma total: f(x0) + 3*Σf(resto) + 2*Σf(múltiplos de 3) + f(xn) = {total_suma:.6f}")

                    print(f"\nResultados Regla de Simpson ({metodo_nombre}):")
                    
                    integral_simbolica = sp.integrate(f_expr, x)
                    print(f"\nFunción original: f(x) = {f_expr}")
                    print(f"Integral real de la función original: F(x) = {integral_simbolica}")
                    
                    print(f"\nLímites de integración: [{a}, {b}]")
                    print(f"Tamaño de paso (h): {h:.6g}")
                    print(f"\nValor Aproximado: {aprox:.6f}")
                    print(f"Valor Exacto: {exacto:.6f}")
                    print(f"Error Absoluto: {error:.6g}")
                    
                    puntos_x = [row['x'] for row in tabla]
                    puntos_y = [row['f(x)'] for row in tabla]
                    
                    puntos_ggb = list(zip(puntos_x, puntos_y))
                    if sub_opcion in ['1', '3']:
                        nombre_archivo = f"integral_simpson_{'1_3' if sub_opcion=='1' else '3_8'}_simple"
                    else:
                        nombre_archivo = f"integral_simpson_{'1_3' if sub_opcion=='2' else '3_8'}_compuesta_n_{n}"
                        
                    crearGgb([f_expr], puntos_ggb, nombre_archivo)
                    
                    graficarIntegracion(f_expr, x, a, b, puntos_x, puntos_y, metodo=metodo_nombre)

            elif opcion == '3':
                print("\n--- Regla de Boole ---")
                print("1. Boole Simple")
                print("2. Boole Compuesta")
                sub_opcion = input("\nElija una opción: ")
                
                if sub_opcion not in ['1', '2']:
                    print("Opción no válida.")
                    continue
                
                expr_str = input("Ingrese la función f(x) ----> ejemplo de sintaxis: exp(x), sin(x):  ")
                if expr_str.strip():
                    f_expr = sp.parse_expr(expr_str)
                else:
                    num_puntos = int(input("Ingrese la cantidad de puntos: "))
                    puntos_x_input = []
                    puntos_y_input = []
                    for i in range(num_puntos):
                        px = float(input(f"x[{i}]: "))
                        py = float(input(f"y[{i}]: "))
                        puntos_x_input.append(px)
                        puntos_y_input.append(py)
                    
                    f_expr = None
                    try:
                        polinomio_lag, _, error_lag = lagrange(puntos_x_input, puntos_y_input, x)
                        if error_lag is None and polinomio_lag is not None:
                            f_expr = polinomio_lag
                            print(f"\nPolinomio construido por Lagrange: f(x) = {f_expr}")
                    except Exception:
                        pass
                    
                    if f_expr is None:
                        try:
                            polinomio_new, _, error_new = newton(puntos_x_input, puntos_y_input, x, 1)
                            if error_new is None and polinomio_new is not None:
                                f_expr = polinomio_new
                                print(f"\nPolinomio construido por Newton: f(x) = {f_expr}")
                        except Exception:
                            pass
                    
                    if f_expr is None:
                        print("Error: No se pudo construir un polinomio con los puntos ingresados.")
                        continue
                
                a = float(input("Ingrese el límite inferior (a): "))
                b = float(input("Ingrese el límite superior (b): "))
                
                if sub_opcion == '1':
                    aprox, exacto, error, tabla, h, error_msg = booleSimple(f_expr, x, a, b)
                    metodo_nombre = "Boole Simple"
                elif sub_opcion == '2':
                    n = int(input("Ingrese el número de subintervalos (n) [debe ser múltiplo de 4]: "))
                    aprox, exacto, error, tabla, h, error_msg = booleCompuesta(f_expr, x, a, b, n)
                    metodo_nombre = "Boole Compuesta"
                    
                if error_msg:
                    print(error_msg)
                else:
                    print("\nTabla de puntos evaluados:")
                    headers = ["i", "xi", "f(xi)"]
                    data = [[row['i'], f"{row['x']:.6f}", f"{row['f(x)']:.6f}"] for row in tabla]
                    
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    f0 = tabla[0]['f(x)']
                    fn = tabla[-1]['f(x)']
                    
                    if sub_opcion == '1':
                        print("\nDesglose del cálculo:")
                        f1, f2, f3 = tabla[1]['f(x)'], tabla[2]['f(x)'], tabla[3]['f(x)']
                        print(f"32 * f(x1) = {32 * f1:.6f}")
                        print(f"12 * f(x2) = {12 * f2:.6f}")
                        print(f"32 * f(x3) = {32 * f3:.6f}")
                        
                        total_suma = 7*f0 + 32*f1 + 12*f2 + 32*f3 + 7*fn
                        print(f"\nSuma total: 7*f(x0) + 32*f(x1) + 12*f(x2) + 32*f(x3) + 7*f(x4) = {total_suma:.6f}")
                    elif sub_opcion == '2':
                        suma_impares = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 2 != 0)
                        suma_pares_no_m4 = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 2 == 0 and row['i'] % 4 != 0)
                        suma_m4 = sum(row['f(x)'] for row in tabla[1:-1] if row['i'] % 4 == 0)
                        
                        print("\nDesglose del cálculo:")
                        print(f"Σf(impares) = {suma_impares:.6f}")
                        print(f"32 * Σf(impares) = {32 * suma_impares:.6f}")
                        print(f"Σf(pares no múltiplos de 4) = {suma_pares_no_m4:.6f}")
                        print(f"12 * Σf(pares no múltiplos de 4) = {12 * suma_pares_no_m4:.6f}")
                        print(f"Σf(múltiplos de 4) = {suma_m4:.6f}")
                        print(f"14 * Σf(múltiplos de 4) = {14 * suma_m4:.6f}")
                        
                        total_suma = 7*f0 + 32*suma_impares + 12*suma_pares_no_m4 + 14*suma_m4 + 7*fn
                        print(f"\nSuma total: 7*f(x0) + 32*Σf(impares) + 12*Σf(pares no m4) + 14*Σf(múltiplos de 4) + 7*f(xn) = {total_suma:.6f}")

                    print(f"\nResultados Regla de Boole ({metodo_nombre}):")
                    
                    integral_simbolica = sp.integrate(f_expr, x)
                    print(f"\nFunción original: f(x) = {f_expr}")
                    print(f"Integral real de la función original: F(x) = {integral_simbolica}")
                    
                    print(f"\nLímites de integración: [{a}, {b}]")
                    print(f"Tamaño de paso (h): {h:.6g}")
                    print(f"\nValor Aproximado: {aprox:.6f}")
                    print(f"Valor Exacto: {exacto:.6f}")
                    print(f"Error Absoluto: {error:.6g}")
                    
                    puntos_x = [row['x'] for row in tabla]
                    puntos_y = [row['f(x)'] for row in tabla]
                    
                    puntos_ggb = list(zip(puntos_x, puntos_y))
                    nombre_archivo = "integral_boole_simple" if sub_opcion == '1' else f"integral_boole_compuesta_n_{n}"
                    crearGgb([f_expr], puntos_ggb, nombre_archivo)
                    
                    graficarIntegracion(f_expr, x, a, b, puntos_x, puntos_y, metodo=metodo_nombre)
                    
        except Exception as e:
            print(f"Error al procesar los datos: {e}")

def menuEcuacionDiferencialNumerica():
    x, y = sp.symbols('x y')
    
    while True:
        print("\n--- MÉTODOS DE ECUACIÓN DIFERENCIAL NUMÉRICA POR BRAYAN ZIPA ---")
        print("\nSintaxis recomendada: sqrt(x), log(x), exp(x), sin(x), cos(x), x**n, x*y")
        print("\n--- Seleccione el método a utilizar ---\n")
        print("1. Método de Euler")
        print("2. Volver al menú principal")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '2':
            break
            
        if opcion not in ['1']:
            print("Opción no válida.")
            continue

        try:
            if opcion == '1':
                print("\n--- Método de Euler ---")
                expr_str = input("Ingrese la función f(x, y) (es decir, y') ----> ejemplo: x - y, x**2 + y: ")
                f_expr = sp.parse_expr(expr_str)
                
                x0 = float(input("Ingrese el valor inicial de x (x0): "))
                xf = float(input("Ingrese el valor final de x (xf): "))
                y0 = float(input("Ingrese el valor inicial de y (y0): "))
                n = int(input("Ingrese la cantidad de pasos (n): "))
                
                h = (xf - x0) / n
                
                exact_str = input("Ingrese la solución exacta y(x) [Deje en blanco para intentar calcularla automáticamente]: ")
                
                exact_expr = None
                if exact_str.strip():
                    exact_expr = sp.parse_expr(exact_str)
                else:
                    print("\nIntentando calcular la solución exacta automáticamente con SymPy...")
                    try:
                        y_func = sp.Function('y')
                        ode_expr = f_expr.subs(y, y_func(x))
                        eq = sp.Eq(y_func(x).diff(x), ode_expr)
                        sol = sp.dsolve(eq, y_func(x), ics={y_func(x0): y0})
                        exact_expr = sol.rhs
                        print(f"Solución exacta calculada: y(x) = {exact_expr}")
                    except Exception as e:
                        print(f"No se pudo calcular la solución exacta de forma analítica: {e}")
                        print("Se continuará solo con la aproximación numérica.")
                
                tabla = euler(f_expr, x, y, x0, y0, xf, h, exact_expr)
                
                print("\nResultados Método de Euler:")
                if exact_expr:
                    headers = ["n", "xi", "yi", "Valor Real", "Error Absoluto", "Error Relativo"]
                    data = [[row['n'], f"{row['xi']:.6g}", f"{row['yi']:.6g}", f"{row['valor_real']:.6g}", f"{row['error_abs']:.6g}", f"{row['error_rel']:.6g}"] for row in tabla]
                else:
                    headers = ["n", "xi", "yi"]
                    data = [[row['n'], f"{row['xi']:.6g}", f"{row['yi']:.6g}"] for row in tabla]
                    
                print(tabulate(data, headers=headers, tablefmt="grid"))
                
                ultimo = tabla[-1]
                print(f"\nResumen de resultados en x = {xf}:")
                print(f"Tamaño de paso (h): {h:.6g}")
                print(f"Valor obtenido (yi): {ultimo['yi']:.6f}")
                if exact_expr:
                    print(f"Valor real: {ultimo['valor_real']:.6f}")
                    print(f"Error absoluto: {ultimo['error_abs']:.6g}")
                    print(f"Error relativo: {ultimo['error_rel']:.6g}")
                    
                # Generar gráfica y archivo GGB
                puntos_x = [row['xi'] for row in tabla]
                puntos_y = [row['yi'] for row in tabla]
                puntos_ggb = list(zip(puntos_x, puntos_y))
                
                exprs = [f_expr]
                if exact_expr:
                    exprs.append(exact_expr)
                    
                    # Intentar obtener polinomio interpolador de los puntos obtenidos por Euler
                    try:
                        pol_interpolado = None
                        
                        # Aplicando método de interpolación de Lagrange
                        try:
                            pol, _, error = lagrange(puntos_x, puntos_y, x)
                            if not error:
                                pol_interpolado = pol
                        except:
                            pass
                        
                        # Si falla, se aplica el método de interpolación de Newton
                        if pol_interpolado is None:
                            try:
                                pol, _, error = newton(puntos_x, puntos_y, x, 1)
                                if not error:
                                    pol_interpolado = pol
                            except:
                                pass
                        
                        if pol_interpolado is not None:
                            exprs.append(pol_interpolado)
                    except Exception as e:
                        print(f"No se pudo generar el polinomio interpolador: {e}")
                        
                crearGgb(exprs, puntos_ggb, f"ec_diferencial_euler_n_{n}")
                graficarEcuacionDiferencial(exact_expr, x, puntos_x, puntos_y, metodo="Euler")
                
        except Exception as e:
            print(f"Error al procesar los datos: {e}")

def menu():
    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Métodos Numéricos")
        print("2. Interpolación")
        print("3. Derivación Numérica")
        print("4. Integración Numérica")
        print("5. Ecuación Diferencial Numérica")
        print("6. Salir")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '1':
            menuMetodos()
        elif opcion == '2':
            menuInterpolacion()
        elif opcion == '3':
            menuDerivacionNumerica()
        elif opcion == '4':
            menuIntegracionNumerica()
        elif opcion == '5':
            menuEcuacionDiferencialNumerica()
        elif opcion == '6':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu()

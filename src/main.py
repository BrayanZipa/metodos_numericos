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
from graficar import graficarMetodos, graficarInterpolacion
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
            print("Saliendo del programa...")
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
                tipo_diferencia = int(input("\nElija que tipo de diferencia dividida calcular: "))

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
    
        except Exception as e:
            print(f"Error al procesar los datos: {e}")

def menu():
    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Métodos Numéricos")
        print("2. Interpolación")
        print("3. Salir")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '1':
            menuMetodos()
        elif opcion == '2':
            menuInterpolacion()
        elif opcion == '3':
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    menu()

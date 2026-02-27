import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tabulate import tabulate
from metodos.metodoBiseccion import biseccion
from metodos.metodoNewton import newtonRaphson
from metodos.metodoSecante import secante
from metodos.metodoPuntoFijo import puntoFijo

def graficar(f_expr, x_sym, raiz, a=None, b=None):
    f_np = sp.lambdify(x_sym, f_expr, 'numpy')
    
    # Determinar rango de la gráfica
    if a is not None and b is not None:
        margin = max(abs(b - a) * 0.5, 1.0)
        x_vals = np.linspace(a - margin, b + margin, 400)
    else:
        x_vals = np.linspace(raiz - 2, raiz + 2, 400)

    y_vals = f_np(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"f(x) = {f_expr}")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter([raiz], [f_np(raiz)], color='red', label=f'Raíz aprox: {raiz:.6f}')
    
    plt.title("Gráfica de la función y su raíz")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.legend()
    plt.grid(True)
    plt.show()

def menu():
    x = sp.symbols('x')
    
    while True:
        print("\n--- MÉTODOS NUMÉRICOS POR BRAYAN ZIPA ---")
        print("\n--- Seleccione el método a utilizar ---\n")
        print("1. Método de Bisección")
        print("2. Método de Newton-Raphson")
        print("3. Método de la Secante")
        print("4. Método de Punto Fijo")
        print("5. Salir")
        
        opcion = input("\nElija una opción: ")
        
        if opcion == '5':
            print("Saliendo del programa...")
            break
            
        if opcion not in ['1', '2', '3', '4']:
            print("Opción no válida.")
            continue

        try:
            if opcion == '4':
                expr_str = input("Ingrese la función g(x) tal que x = g(x) (ej: cos(x)): ")
            else:
                expr_str = input("Ingrese la función f(x) (ej: x**2 - 4 o cos(x) - x): ")
            
            f_expr = sp.parse_expr(expr_str)
            tol = float(input("Ingrese la tolerancia (ej: 0.01): "))
            
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
                    graficar(f_expr, x, raiz_final, a, b)

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
                    graficar(f_expr, x, raiz_final)
            
            elif opcion == '3':
                x0 = float(input("Ingrese el valor inicial (x0): "))
                x1 = float(input("Ingrese el valor inicial (x1): "))
                
                iteraciones, error_msg = secante(f_expr, x, x0, x1, tol)
                
                if error_msg:
                    print(error_msg)
                else:
                    print("\nResultados Método de la Secante:")
                    headers = ["Iter", "x0", "x1", "f(x0)", "f(x1)", "x_nuevo", "Error"]
                    data = [[i['iter'], f"{i['x0']:.6f}", f"{i['x1']:.6f}", f"{i['f(x0)']:.6f}", f"{i['f(x1)']:.6f}", f"{i['xi+1']:.6f}", f"{i['error']:.6f}"] for i in iteraciones]
                    print(tabulate(data, headers=headers, tablefmt="grid"))
                    
                    raiz_final = iteraciones[-1]['xi+1']
                    graficar(f_expr, x, raiz_final)
            
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
                    graficar(f_expr, x, raiz_final)

        except Exception as e:
            print(f"Error al procesar los datos: {e}")

if __name__ == "__main__":
    menu()

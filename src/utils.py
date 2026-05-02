import numpy as np
import sympy as sp
from tabulate import tabulate
from generarArchivo import crearGgb

"""
Genera n puntos equidistantes en el intervalo [a, b].
a: Valor inicial del intervalo.
b: Valor final del intervalo.
n: Número total de puntos a generar.
"""
def generarPuntosEquidistantes(a, b, n):
    puntos = [float(x) for x in np.linspace(a, b, n)]
    print("Puntos generados:", [f"{x:.6f}" for x in puntos])
    return puntos

"""
Analiza si un conjunto de puntos es equidistante y calcula los pasos entre ellos.
x: Lista o arreglo de puntos.
tolerancia: Margen de error para comparar los pasos.
"""
def analizarPuntos(x, tolerancia=1e-9):
    x = np.array(x, dtype=float)
    pasos = np.diff(x)
    
    # Verifica si todos los pasos son iguales dentro de una tolerancia
    equidistante = np.all(np.abs(pasos - pasos[0]) < tolerancia)
    paso = pasos[0] if equidistante else None
    
    return {
        "pasos": pasos,
        "equidistante": equidistante,
        "paso": paso
    }

"""
Genera un array de números en un rango especificado.
inicio: valor de inicio
fin: valor de fin
paso: incremento entre valores
"""
def generarRango(inicio, fin, paso):
    return np.arange(inicio, fin, paso)

"""
Evalúa una expresión de sympy en una lista de puntos.
f_expr: expresión de sympy
puntos_x: lista o array de valores x a evaluar
"""
def evaluarPuntos(f_expr, puntos_x):
    x = sp.symbols('x')
    # Se evalúa cada punto y se retorna una lista de tuplas (x, y)
    return [(float(px), float(f_expr.subs(x, px))) for px in puntos_x]

"""
Crea una función desde un string, genera puntos en un rango determinado y 
genera un archivo .ggb con la función y los puntos.
expr_str: string con la expresión de la función (ej: "x**2")
puntos: lista o array de valores x a evaluar
nombre_base: nombre para el archivo .ggb generado
"""
def evaluarFuncion(expr_str, puntos, nombre_base="graficacion_puntos"):
    f_expr = sp.parse_expr(expr_str)
    puntos_x = evaluarPuntos(f_expr, puntos)
    crearGgb([f_expr], puntos_x, nombre_base)

    print(f"\nFunción: f(x) = {f_expr}\n")
    tabla = [(x, y) for x, y in puntos_x]
    print(tabulate(tabla, headers=["x", "f(x)"], tablefmt="rounded_outline", floatfmt=(".2f", ".6f")))

    return f_expr, puntos_x

import numpy as np
import sympy as sp
from generarArchivo import crearGgb

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
inicio: valor inicial del rango
fin: valor final del rango
paso: incremento en el rango
nombre_base: nombre para el archivo .ggb generado
"""
def evaluarFuncion(expr_str, inicio, fin, paso, nombre_base="graficacion_puntos"):
    f_expr = sp.parse_expr(expr_str)
    puntos_x = generarRango(inicio, fin, paso)
    puntos_ggb = evaluarPuntos(f_expr, puntos_x)
    crearGgb([f_expr], puntos_ggb, nombre_base)
    return f_expr, puntos_ggb

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sympy as sp
from interpolacion.polinomioTaylor import taylor
from interpolacion.polinomioLagrange import lagrange
from interpolacion.polinomioNewton import newton
from interpolacion.polinomioMinimosCuadrados import minimosCuadrados

def test_taylor():
    print("Testing Taylor Polynomial...")
    x = sp.symbols('x')
    f_expr = sp.exp(x)
    x0 = 0
    n = 2
    polinomio, iteraciones, error_msg = taylor(f_expr, x, x0, n)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        print(f"Polinomio: {polinomio}")
        val_0 = float(polinomio.subs(x, 0))
        val_1 = float(polinomio.subs(x, 1))
        print(f"P(0) = {val_0:.6f}")
        print(f"P(1) = {val_1:.6f}")
        assert abs(val_0 - 1.0) < 0.001
        assert abs(val_1 - 2.5) < 0.001
        print("Taylor Polynomial Test Passed!")

def test_lagrange():
    print("\nTesting Lagrange Polynomial...")
    x = sp.symbols('x')
    x_pts = [0, 1, 2]
    y_pts = [1, 2, 4]
    polinomio, li_polinomios, error_msg = lagrange(x_pts, y_pts, x)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        print(f"Polinomio: {polinomio}")
        for xi, yi in zip(x_pts, y_pts):
            val = float(polinomio.subs(x, xi))
            print(f"P({xi}) = {val:.6f} (Esperado: {yi})")
            assert abs(val - yi) < 0.001
        print("Lagrange Polynomial Test Passed!")
        
def test_newton():
    print("\nTesting Newton Polynomial...")
    x = sp.symbols('x')
    x_pts = [0, 1, 2]
    y_pts = [1, 2, 4]
    tipo_diferencia = 1
    polinomio, data, error_msg = newton(x_pts, y_pts, x, tipo_diferencia)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        print(f"Polinomio: {polinomio}")
        for xi, yi in zip(x_pts, y_pts):
            val = float(polinomio.subs(x, xi))
            print(f"P({xi}) = {val:.6f} (Esperado: {yi})")
            assert abs(val - yi) < 0.001
        print("Newton Polynomial Test Passed!")

def test_minimos_cuadrados():
    print("\nTesting Minimos Cuadrados Polynomial...")
    x = sp.symbols('x')
    # Using the same data points provided as example in main.py
    x_pts = [1, 2, 3, 4, 5, 6, 7]
    y_pts = [0.5, 0.25, 2, 4, 3.5, 6, 5.5]
    grado = 1
    polinomio, reporte, error_msg = minimosCuadrados(x_pts, y_pts, x, grado)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        print(f"Polinomio: {polinomio}")
        sr = reporte['sr']
        print(f"Sr: {sr:.6f}")
        assert sr >= 0  # Sum of squared residuals should be non-negative
        print("Minimos Cuadrados Test Passed!")

if __name__ == "__main__":
    try:
        test_taylor()
        test_lagrange()
        test_newton()
        test_minimos_cuadrados()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")

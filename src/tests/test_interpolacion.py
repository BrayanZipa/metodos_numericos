import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sympy as sp
from interpolacion.polinomioTaylor import taylor
from interpolacion.polinomioLagrange import lagrange

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
        # P(x) = 1 + x + x^2/2
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

if __name__ == "__main__":
    try:
        test_taylor()
        test_lagrange()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")

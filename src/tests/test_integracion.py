import sympy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integracion.reglaTrapecio import trapecioSimple, trapecioCompuesta
from integracion.reglaSimpson import simpsonUnTercioSimple, simpsonUnTercioCompuesta, simpsonTresOctavosSimple, simpsonTresOctavosCompuesta
from integracion.reglaBoole import booleSimple, booleCompuesta

def test_trapecio_simple():
    print("Testing Trapecio Simple...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    
    aprox, exacto, error, tabla, h, msg = trapecioSimple(f_expr, x, a, b)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        # Trapecio simple: h=2. I = (2/2) * (0 + 4) = 4
        assert abs(aprox - 4.0) < 1e-6
        print("Trapecio Simple Test Passed!")

def test_trapecio_compuesta():
    print("\nTesting Trapecio Compuesta...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    n = 2
    
    aprox, exacto, error, tabla, h, msg = trapecioCompuesta(f_expr, x, a, b, n)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        # Trapecio compuesta (n=2): h=1. I = (1/2) * (0 + 2*(1) + 4) = 3
        assert abs(aprox - 3.0) < 1e-6
        print("Trapecio Compuesta Test Passed!")

def test_simpson_1_3_simple():
    print("\nTesting Simpson 1/3 Simple...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    
    aprox, exacto, error, tabla, h, msg = simpsonUnTercioSimple(f_expr, x, a, b)
    
    if msg:
        print(f"Error: {msg}")
    else:
        # Simpson 1/3 es exacto para polinomios de grado <= 2
        assert abs(exacto - 2.6666666666666665) < 1e-6
        assert abs(aprox - 2.6666666666666665) < 1e-6
        print("Simpson 1/3 Simple Test Passed!")

def test_simpson_1_3_compuesta():
    print("\nTesting Simpson 1/3 Compuesta...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    n = 2
    
    aprox, exacto, error, tabla, h, msg = simpsonUnTercioCompuesta(f_expr, x, a, b, n)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        assert abs(aprox - 2.6666666666666665) < 1e-6
        print("Simpson 1/3 Compuesta Test Passed!")

def test_simpson_3_8_simple():
    print("\nTesting Simpson 3/8 Simple...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    
    aprox, exacto, error, tabla, h, msg = simpsonTresOctavosSimple(f_expr, x, a, b)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        assert abs(aprox - 2.6666666666666665) < 1e-6
        print("Simpson 3/8 Simple Test Passed!")

def test_simpson_3_8_compuesta():
    print("\nTesting Simpson 3/8 Compuesta...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    n = 3
    
    aprox, exacto, error, tabla, h, msg = simpsonTresOctavosCompuesta(f_expr, x, a, b, n)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        assert abs(aprox - 2.6666666666666665) < 1e-6
        print("Simpson 3/8 Compuesta Test Passed!")

def test_boole_simple():
    print("\nTesting Boole Simple...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    
    aprox, exacto, error, tabla, h, msg = booleSimple(f_expr, x, a, b)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        assert abs(aprox - 2.6666666666666665) < 1e-6
        print("Boole Simple Test Passed!")

def test_boole_compuesta():
    print("\nTesting Boole Compuesta...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    a = 0
    b = 2
    n = 4
    
    aprox, exacto, error, tabla, h, msg = booleCompuesta(f_expr, x, a, b, n)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 2.6666666666666665) < 1e-6
        assert abs(aprox - 2.6666666666666665) < 1e-6
        print("Boole Compuesta Test Passed!")

if __name__ == "__main__":
    try:
        test_trapecio_simple()
        test_trapecio_compuesta()
        test_simpson_1_3_simple()
        test_simpson_1_3_compuesta()
        test_simpson_3_8_simple()
        test_simpson_3_8_compuesta()
        test_boole_simple()
        test_boole_compuesta()
        print("\nAll integration tests passed successfully!")
    except AssertionError as e:
        print(f"\nTests failed: AssertionError. Check your calculations.")
    except Exception as e:
        print(f"\nTests failed: {e}")

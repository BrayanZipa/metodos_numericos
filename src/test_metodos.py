import sympy as sp
from metodos.metodoSecante import secante
from metodos.metodoPuntoFijo import puntoFijo
from metodos.metodoNewton import newtonRaphson
from metodos.metodoBiseccion import biseccion

def test_biseccion():
    print("Testing Bisection Method...")
    x = sp.symbols('x')
    f_expr = x**2 - 4
    a, b = 1.0, 3.0
    tol = 0.0001
    iteraciones, error_msg = biseccion(f_expr, x, a, b, tol)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        raiz = iteraciones[-1]['pn']
        print(f"Raíz encontrada: {raiz:.6f}")
        assert abs(raiz - 2.0) < 0.1
        print("Bisection Method Test Passed!")

def test_newton():
    print("\nTesting Newton-Raphson Method...")
    x = sp.symbols('x')
    f_expr = x**2 - 4
    x0 = 1.0
    tol = 0.0001
    iteraciones, error_msg = newtonRaphson(f_expr, x, x0, tol)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        raiz = iteraciones[-1]['xi+1']
        print(f"Raíz encontrada: {raiz:.6f}")
        assert abs(raiz - 2.0) < 0.1
        print("Newton-Raphson Method Test Passed!")

def test_secante():
    print("\nTesting Secant Method...")
    x = sp.symbols('x')
    f_expr = x**2 - 4
    x0, x1 = 1.0, 3.0
    tol = 0.0001
    iteraciones, error_msg = secante(f_expr, x, x0, x1, tol)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        raiz = iteraciones[-1]['xi+1']
        print(f"Raíz encontrada: {raiz:.6f}")
        assert abs(raiz - 2.0) < 0.1
        print("Secant Method Test Passed!")

def test_punto_fijo():
    print("\nTesting Fixed Point Method...")
    x = sp.symbols('x')
    # x = cos(x)
    g_expr = sp.cos(x)
    x0 = 0.5
    tol = 0.0001
    iteraciones, error_msg = puntoFijo(g_expr, x, x0, tol)
    
    if error_msg:
        print(f"Error: {error_msg}")
    else:
        raiz = iteraciones[-1]['g(xi)']
        print(f"Raíz encontrada: {raiz:.6f}")
        # La raíz de cos(x) = x es aproximadamente 0.739085
        assert abs(raiz - 0.739085) < 0.1
        print("Fixed Point Method Test Passed!")

if __name__ == "__main__":
    try:
        test_biseccion()
        test_newton()
        test_secante()
        test_punto_fijo()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")

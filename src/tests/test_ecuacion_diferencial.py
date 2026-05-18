import sympy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ecuacionDiferencial.metodoEuler import euler
from ecuacionDiferencial.metodoRungeKutta import rungeKuttaSegundoOrden, rungeKuttaCuartoOrden

def test_euler():
    print("Testing Método de Euler...")
    x, y = sp.symbols('x y')
    # Ecuación de prueba: y' = x + y, y(0) = 1
    f_expr = sp.parse_expr("x + y")
    x0 = 0
    y0 = 1
    xf = 0.5
    h = 0.1
    # Solución exacta: y(x) = 2*e^x - x - 1
    exact_expr = sp.parse_expr("2*exp(x) - x - 1")
    
    tabla = euler(f_expr, x, y, x0, y0, xf, h, exact_expr)
    
    assert tabla is not None
    assert len(tabla) == 6 # n = 5 + 1 inicial
    
    last_row = tabla[-1]
    
    # Validamos que el último x es aproximadamente 0.5
    assert abs(last_row['xi'] - 0.5) < 1e-6
    
    print(f"  Último y aprox: {last_row['yi']:.6f}, Exacto: {last_row['valor_real']:.6f}, Error rel: {last_row['error_rel']:.6g}")
    print("Método de Euler Test Passed!")

def test_runge_kutta_segundo_orden():
    print("Testing Método de Runge-Kutta 2do Orden...")
    x, y = sp.symbols('x y')
    f_expr = sp.parse_expr("x + y")
    x0 = 0
    y0 = 1
    xf = 0.5
    h = 0.1
    exact_expr = sp.parse_expr("2*exp(x) - x - 1")
    
    tabla = rungeKuttaSegundoOrden(f_expr, x, y, x0, y0, xf, h, exact_expr)
    
    assert tabla is not None
    assert len(tabla) == 6
    
    last_row = tabla[-1]
    assert abs(last_row['xi'] - 0.5) < 1e-6
    
    print(f"  Último y aprox: {last_row['yi']:.6f}, Exacto: {last_row['valor_real']:.6f}, Error rel: {last_row['error_rel']:.6g}")
    print("Método de Runge-Kutta 2do Orden Test Passed!")

def test_runge_kutta_cuarto_orden():
    print("Testing Método de Runge-Kutta 4to Orden...")
    x, y = sp.symbols('x y')
    f_expr = sp.parse_expr("x + y")
    x0 = 0
    y0 = 1
    xf = 0.5
    h = 0.1
    exact_expr = sp.parse_expr("2*exp(x) - x - 1")
    
    tabla = rungeKuttaCuartoOrden(f_expr, x, y, x0, y0, xf, h, exact_expr)
    
    assert tabla is not None
    assert len(tabla) == 6
    
    last_row = tabla[-1]
    assert abs(last_row['xi'] - 0.5) < 1e-6
    
    print(f"  Último y aprox: {last_row['yi']:.6f}, Exacto: {last_row['valor_real']:.6f}, Error rel: {last_row['error_rel']:.6g}")
    print("Método de Runge-Kutta 4to Orden Test Passed!")

if __name__ == "__main__":
    try:
        test_euler()
        test_runge_kutta_segundo_orden()
        test_runge_kutta_cuarto_orden()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")

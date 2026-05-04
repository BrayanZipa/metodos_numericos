import sympy as sp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from derivacion.derivacionNumerica import derivadaDosPuntos, derivadaTresPuntos, derivadaCincoPuntos, derivadaOrdenSuperior

def test_derivada_dos_puntos():
    print("Testing Derivada de 2 Puntos...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    x0 = 2
    h = 0.1
    
    aprox, exacto, error, error_teo, tabla, h_ret, nombre, msg = derivadaDosPuntos(f_expr, x, x0, h, 1)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 4.0) < 1e-6
        assert abs(aprox - 4.1) < 1e-6
        assert tabla is not None and len(tabla) == 2
        assert error_teo is not None
        print(f"  Aprox: {aprox:.6f}, Exacto: {exacto:.6f}, Error: {error:.6g}, Error Teórico: {error_teo:.6g}")
        print(f"  Puntos utilizados: {len(tabla)}")
        print("Derivada de 2 Puntos Test Passed!")

def test_derivada_tres_puntos():
    print("Testing Derivada de 3 Puntos...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("sin(x)")
    x0 = 1.0
    h = 0.1
    
    aprox, exacto, error, error_teo, tabla, h_ret, nombre, msg = derivadaTresPuntos(f_expr, x, x0, h, 3)
    
    if msg:
        print(f"Error: {msg}")
    else:
        # cos(1) ≈ 0.540302
        assert abs(exacto - 0.540302) < 1e-3
        assert error < 0.01
        assert tabla is not None and len(tabla) >= 2
        assert error_teo is not None
        print(f"  Aprox: {aprox:.6f}, Exacto: {exacto:.6f}, Error: {error:.6g}, Error Teórico: {error_teo:.6g}")
        print(f"  Puntos utilizados: {len(tabla)}")
        print("Derivada de 3 Puntos Test Passed!")

def test_derivada_cinco_puntos():
    print("Testing Derivada de 5 Puntos...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("exp(x)")
    x0 = 0
    h = 0.1
    
    aprox, exacto, error, error_teo, tabla, h_ret, nombre, msg = derivadaCincoPuntos(f_expr, x, x0, h, 1)
    
    if msg:
        print(f"Error: {msg}")
    else:
        # exp'(0) = 1
        assert abs(exacto - 1.0) < 1e-6
        assert error < 1e-4
        assert tabla is not None and len(tabla) >= 4
        assert error_teo is not None
        print(f"  Aprox: {aprox:.6f}, Exacto: {exacto:.6f}, Error: {error:.6g}, Error Teórico: {error_teo:.6g}")
        print(f"  Puntos utilizados: {len(tabla)}")
        print("Derivada de 5 Puntos Test Passed!")

def test_derivada_orden_superior():
    print("Testing Derivada de Orden Superior...")
    x = sp.symbols('x')
    # f(x) = x^4, f'''(x) = 24x. Evaluada en x0=2 -> exacta = 48
    f_expr = sp.parse_expr("x**4")
    x0 = 2
    h = 0.1
    
    aprox, exacto, error, tabla, h_ret, nombre, msg = derivadaOrdenSuperior(f_expr, x, x0, h, orden=3, tipo_diferencia=3)
    
    if msg:
        print(f"Error: {msg}")
    else:
        assert abs(exacto - 48.0) < 1e-6
        assert error < 0.5
        assert tabla is not None and len(tabla) >= 2
        print(f"  Aprox: {aprox:.6f}, Exacto: {exacto:.6f}, Error: {error:.6g}")
        print(f"  Puntos utilizados: {len(tabla)}")
        print("Derivada de Orden Superior Test Passed!")

def test_derivada_tipo_invalido():
    print("Testing Derivada con Tipo Inválido...")
    x = sp.symbols('x')
    f_expr = sp.parse_expr("x**2")
    
    _, _, _, _, _, _, _, msg = derivadaDosPuntos(f_expr, x, 2, 0.1, 4)
    
    assert msg is not None
    assert "Tipo de diferencia no válido" in msg
    print("Derivada con Tipo Inválido Test Passed!")

if __name__ == "__main__":
    try:
        test_derivada_dos_puntos()
        test_derivada_tres_puntos()
        test_derivada_cinco_puntos()
        test_derivada_orden_superior()
        test_derivada_tipo_invalido()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")

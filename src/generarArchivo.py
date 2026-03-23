import os
import re
import zipfile
from datetime import datetime

"""
Convierte un número a cadena evitando la notación científica 'e' si es posible, o formateándola a (*10^).
val: valor numérico a formatear
"""
def formatNum(val):
    if isinstance(val, (int, float)):
        # Si es muy pequeño, se usa más precisión para evitar que se convierta en 0
        s = f"{val:.15f}".rstrip('0').rstrip('.')
        if s == "0" or s == "-0":
             return f"{val:.20g}".replace('e', '*10^(') + ')' if 'e' in f"{val:.20g}" else f"{val:.20g}"
        return s
    return str(val)

"""
Genera la estructura XML para un conjunto de puntos.
puntos: lista de tuplas (x, y)
"""
def generarPuntos(puntos):
    xml = ""
    letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, (x_val, y_val) in enumerate(puntos):
        if i < len(letras):
            label = letras[i]
        else:
            letra = letras[i % len(letras)]
            numero = i // len(letras)
            label = f"{letra}{numero}"

        # Formatear para XML (coordenadas numéricas puras)
        x_str = formatNum(x_val)
        y_str = formatNum(y_val)
        
        # Evitar notación científica y usar formato decimal para coords
        x_xml = f"{float(x_val):.15f}".rstrip('0').rstrip('.')
        y_xml = f"{float(y_val):.15f}".rstrip('0').rstrip('.')

        xml += f"""
        <expression label="{label}" exp="({x_str}, {y_str})" type="point"/>
        <element type="point" label="{label}">
            <show object="true" label="true"/>
            <objColor r="77" g="77" b="255" alpha="0"/>
            <layer val="0"/>
            <labelMode val="0"/>
            <animation step="0.1" type="1" playing="false"/>
            <pointSize val="5"/>
            <pointStyle val="0"/>
            <coords x="{x_xml}" y="{y_xml}" z="1"/>
        </element>
        """
    return xml

"""
Genera la estructura XML para un conjunto de funciones.
Maneja la conversión de notación científica 'e' a el formato (*10^).
funciones: lista de expresiones de sympy o cadenas
"""
def generarFunciones(funciones):
    xml = ""
    letras = ["f", "g", "h", "p", "q", "r"]
    colores = [
        (0, 100, 0),    # verde
        (255, 0, 0),    # rojo
        (0, 0, 255),    # azul
        (255, 165, 0),  # naranja
        (128, 0, 128),  # morado
        (0, 255, 255)   # cyan
    ]

    for i, expresion in enumerate(funciones):
        if i < len(letras):
            label = letras[i]
        else:
            letra = letras[i % len(letras)]
            numero = i // len(letras)
            label = f"{letra}{numero}"
        
        r, g, b = colores[i % len(colores)]    

        # Convertir la expresión a string y manejar notación científica
        expr_str = str(expresion)
        # Reemplaza la nomenclatura 3.8e-08 por 3.8*10^(-08)
        expr_str = re.sub(r"(\d+\.?\d*)e([-+]?\d+)", r"\1*10^(\2)", expr_str)

        xml += f"""
        <expression label="{label}" exp="{label}(x) = {expr_str}" type="function"/>
        <element type="function" label="{label}">
            <show object="true" label="true" ev="4"/>
            <objColor r="{r}" g="{g}" b="{b}" alpha="0"/>
            <layer val="0"/>
            <labelMode val="0"/>
            <fixed val="true"/>
            <lineStyle thickness="5" type="0" typeHidden="1" opacity="178"/>
        </element>
        """
    return xml

"""
Crea el contenido completo del archivo .xml.
funciones_xml: cadena XML con las definiciones de funciones
puntos_xml: cadena XML con las definiciones de puntos
"""
def crearXml(funciones_xml, puntos_xml):
    return f"""<?xml version="1.0" encoding="utf-8"?>
    <geogebra format="5.0" version="5.0.0.0">
        <construction title="Generado con Python" author="" date="">
            {funciones_xml}
            {puntos_xml}
        </construction>
    </geogebra>
    """

"""
Función principal que genera un archivo comprimido .ggb.
funciones: lista de funciones a graficar
puntos: lista de puntos (x, y) a graficar
nombre_base: nombre inicial para el archivo generado
"""
def crearGgb(funciones = None, puntos = None, nombre_base = "archivo"):
    funciones = funciones or []
    puntos = puntos or []

    # Generar un nombre único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_final = f"{nombre_base}_{timestamp}"

    # Crear carpeta si no existe
    carpeta_xml = "archivos_xml"
    carpeta_ggb = "archivos_ggb"
    os.makedirs(carpeta_xml, exist_ok=True)
    os.makedirs(carpeta_ggb, exist_ok=True)

    ruta_xml = os.path.join(carpeta_xml, f"{nombre_final}.xml")
    ruta_ggb = os.path.join(carpeta_ggb, f"{nombre_final}.ggb")

    # Generar estructura del archivo .XML
    funciones_xml = generarFunciones(funciones) if funciones else ""
    puntos_xml = generarPuntos(puntos) if puntos else ""
    xml_final = crearXml(funciones_xml, puntos_xml)

    # Crear archivo .XML
    with open(ruta_xml, "w", encoding="utf-8") as f:
        f.write(xml_final)

    # Crear archivo .GGB
    with zipfile.ZipFile(ruta_ggb, "w") as z:
        z.write(ruta_xml, "geogebra.xml")

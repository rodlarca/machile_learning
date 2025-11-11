# Ejemplo de Implementación con Operadores Sobrecargados

# Importar librerías necesarias
import numpy as np
import pandas as pd

# Crear un DataFrame de ejemplo
data = {
    'Ventas': [100, 150, 200, 250, 300],
    'Costos': [50, 70, 90, 120, 140],
    'Descuentos': [5, 10, 15, 20, 25]
}

df = pd.DataFrame(data)
# Crear nuevas características usando operadores sobrecargados
df['Margen'] = df['Ventas'] - df['Costos']
df['Ratio_Descuento'] = df['Descuentos'] / df['Ventas']
df['Margen_Ajustado'] = df['Margen'] * (1 - df['Ratio_Descuento'])

print(df)

# Ejemplo de Implementación con Manejo de Cadenas

import re

# Crear un DataFrame de ejemplo
data = {'Texto': ['Hola mundo', 'Machine Learning', 'Python es genial', 'Data Science']}
df = pd.DataFrame(data)

# Calcular la longitud de cada cadena
df['Longitud'] = df['Texto'].apply(len)

# Contar la cantidad de palabras en cada cadena
df['Num_Palabras'] = df['Texto'].apply(lambda x: len(x.split()))

# Detectar la presencia de una palabra específica usando expresiones regulares
df['Contiene_Python'] = df['Texto'].apply(lambda x: bool(re.search(r'Python', x)))

print(df)

# Ejemplo de Implementación con Fechas y Horas

import pandas as pd

# Crear un DataFrame de ejemplo con fechas
data = {'Fechas': pd.to_datetime(['2023-01-01', '2023-06-15', '2024-12-31'])}
df = pd.DataFrame(data)

# Extraer el día, mes y año
df['Día'] = df['Fechas'].dt.day
df['Mes'] = df['Fechas'].dt.month
df['Año'] = df['Fechas'].dt.year

# Calcular si la fecha es un fin de semana
df['Es_Fin_de_Semana'] = df['Fechas'].dt.dayofweek >= 5

print(df)

# Ejemplo de Implementación con Funciones Personalizadas

# Crear un DataFrame de ejemplo
data = {'Valores': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Definir una función personalizada para escalar datos
def escalar_datos(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

# Aplicar la función personalizada
df['Valores_Escalados'] = df['Valores'].apply(escalar_datos, args=(df['Valores'].min(), df['Valores'].max()))

print(df)
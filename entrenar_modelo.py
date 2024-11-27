import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Cargar el dataset
df_postulantes = pd.read_excel('/postulantes.xlsx')  # Cambia la ruta del archivo

# Verificar las primeras filas para entender el formato de los datos
print(df_postulantes.head())

# Reemplazar '\\N' por NaN (si se encuentra en varias columnas)
df_postulantes.replace('\\N', pd.NA, inplace=True)

# Limpiar valores no numéricos en la columna 'PUNTAJE' y convertirla a numérico
df_postulantes['PUNTAJE'] = pd.to_numeric(df_postulantes['PUNTAJE'], errors='coerce')

# Limpiar cualquier otra columna numérica si es necesario
columnas_numericas = ['UBIGEO_COLEGIO', 'ANIO_EGRESO_COLEGIO', 'DEPARTAMENTO_COLEGIO', 
                      'PROVINCIA_COLEGIO', 'DISTRITO_COLEGIO']

for col in columnas_numericas:
    df_postulantes[col] = pd.to_numeric(df_postulantes[col], errors='coerce')

# Verificar si hay valores faltantes en el dataset
print(df_postulantes.isna().sum())

# Eliminar filas con NaN en 'PUNTAJE' y 'APTO'
df_postulantes.dropna(subset=['PUNTAJE', 'APTO'], inplace=True)

# Verificar que no haya NaN en 'APTO'
print("Valores NaN en 'APTO' después de la limpieza:", df_postulantes['APTO'].isna().sum())

# Mapear 'SI' -> 1 y 'NO' -> 0 en la columna 'APTO'
df_postulantes['APTO'] = df_postulantes['APTO'].map({'SI': 1, 'NO': 0})

# Codificar las columnas categóricas
label_encoder = LabelEncoder()

df_postulantes['TIPO_COLEGIO'] = label_encoder.fit_transform(df_postulantes['TIPO_COLEGIO'].astype(str))
df_postulantes['COLEGIO_PROCEDENCIA'] = label_encoder.fit_transform(df_postulantes['COLEGIO_PROCEDENCIA'].astype(str))
df_postulantes['UBIGEO_COLEGIO'] = label_encoder.fit_transform(df_postulantes['UBIGEO_COLEGIO'].astype(str))
df_postulantes['DEPARTAMENTO_COLEGIO'] = label_encoder.fit_transform(df_postulantes['DEPARTAMENTO_COLEGIO'].astype(str))
df_postulantes['PROVINCIA_COLEGIO'] = label_encoder.fit_transform(df_postulantes['PROVINCIA_COLEGIO'].astype(str))
df_postulantes['DISTRITO_COLEGIO'] = label_encoder.fit_transform(df_postulantes['DISTRITO_COLEGIO'].astype(str))

# Definir las características (X) y el objetivo (y)
X = df_postulantes[['TIPO_COLEGIO', 'COLEGIO_PROCEDENCIA', 'UBIGEO_COLEGIO', 'ANIO_EGRESO_COLEGIO', 
                    'DEPARTAMENTO_COLEGIO', 'PROVINCIA_COLEGIO', 'DISTRITO_COLEGIO']]
y = df_postulantes['APTO']

# Verificar que no haya NaN en 'y'
print("Valores NaN en 'y' después de la limpieza:", y.isna().sum())

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del modelo: {accuracy * 100:.2f}%')

# Guardar el modelo entrenado
joblib.dump(model, 'modelo_ingreso_universidad.pkl')

print("Modelo entrenado y guardado correctamente.")

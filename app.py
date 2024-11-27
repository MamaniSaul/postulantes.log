from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_ingreso_universidad.pkl')

# Crear un codificador para las variables categóricas (si ya no está procesado, se hace aquí)
le = LabelEncoder()

@app.route('/')
def index():
    # Renderiza la página principal con el formulario
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        # Obtener los datos del formulario (valores seleccionados)
        tipo_colegio = request.form['TIPO_COLEGIO']
        colegio_procedencia = request.form['COLEGIO_PROCEDENCIA']
        ubigeo_colegio = request.form['UBIGEO_COLEGIO']
        anio_egreso_colegio = request.form['ANIO_EGRESO_COLEGIO']
        departamento_colegio = request.form['DEPARTAMENTO_COLEGIO']
        provincia_colegio = request.form['PROVINCIA_COLEGIO']
        distrito_colegio = request.form['DISTRITO_COLEGIO']
        
        # Crear un diccionario con los datos recibidos
        input_data = {
            'TIPO_COLEGIO': [tipo_colegio],
            'COLEGIO_PROCEDENCIA': [colegio_procedencia],
            'UBIGEO_COLEGIO': [ubigeo_colegio],
            'ANIO_EGRESO_COLEGIO': [anio_egreso_colegio],
            'DEPARTAMENTO_COLEGIO': [departamento_colegio],
            'PROVINCIA_COLEGIO': [provincia_colegio],
            'DISTRITO_COLEGIO': [distrito_colegio]
        }

        # Convertir a un DataFrame
        df_input = pd.DataFrame(input_data)

        # Si es necesario, codificar las variables categóricas aquí
        df_input['TIPO_COLEGIO'] = le.fit_transform(df_input['TIPO_COLEGIO'])
        df_input['COLEGIO_PROCEDENCIA'] = le.fit_transform(df_input['COLEGIO_PROCEDENCIA'])
        df_input['DEPARTAMENTO_COLEGIO'] = le.fit_transform(df_input['DEPARTAMENTO_COLEGIO'])
        df_input['PROVINCIA_COLEGIO'] = le.fit_transform(df_input['PROVINCIA_COLEGIO'])
        df_input['DISTRITO_COLEGIO'] = le.fit_transform(df_input['DISTRITO_COLEGIO'])

        # Realizar la predicción con el modelo
        prediccion = modelo.predict(df_input)

        # Mostrar el resultado en la interfaz
        resultado = "Apto" if prediccion[0] == 1 else "No Apto"
        return render_template('index.html', resultado=resultado)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
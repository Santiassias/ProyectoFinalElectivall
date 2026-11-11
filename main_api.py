import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# 1. Cargar el paquete serializado una sola vez al iniciar el servidor
paquete = joblib.load('models/modelo_scouting_rf.joblib')
modelo = paquete['modelo']
columnas_entrenamiento = paquete['columnas_X']

app = FastAPI(title="API de Scouting Deportivo")

# Permite que index.html (cliente) consuma la API desde el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Esquema de entrada validado con Pydantic
class Jugador(BaseModel):
    jugador: Optional[str] = "Candidato"
    posicion: str
    edad: Optional[float] = Field(default=None, ge=16, le=50)            # Rango de edad del dataset
    estatura_cm: Optional[float] = Field(default=None, ge=150, le=210)   # Rango de estatura válido


# 3. Endpoint de predicción
@app.post("/predict")
def predecir(datos: Jugador):
    df = pd.DataFrame([datos.model_dump()])

    # Aplicar las MISMAS reglas de imputación aprendidas en entrenamiento
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')
    df['estatura_cm'] = df['estatura_cm'].fillna(df['posicion'].map(paquete['medianas_estatura']))
    df['estatura_cm'] = df['estatura_cm'].fillna(paquete['mediana_global_estatura'])
    df['edad'] = df['edad'].fillna(paquete['mediana_global_edad'])

    # Variables categóricas alineadas con la matriz original
    df_dummies = pd.get_dummies(df[['posicion']], dtype=int)
    X_nuevo = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    X_nuevo = X_nuevo.reindex(columns=columnas_entrenamiento, fill_value=0)

    # Predicción y probabilidad
    prediccion = int(modelo.predict(X_nuevo)[0])
    probabilidad = float(modelo.predict_proba(X_nuevo)[0, 1])

    return {
        "jugador": datos.jugador,
        "posicion": datos.posicion,
        "edad": float(df['edad'][0]),
        "estatura_cm": float(df['estatura_cm'][0]),
        "prediccion": prediccion,
        "clase": "Élite" if prediccion == 1 else "Estándar",
        "probabilidad_elite": round(probabilidad * 100, 2)
    }


# 4. Servir el aplicativo web desde la misma API (http://127.0.0.1:8000)
@app.get("/")
def inicio():
    return FileResponse('index.html')

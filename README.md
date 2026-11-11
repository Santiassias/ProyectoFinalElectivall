# Ecosistema Analítico End-to-End para Scouting Deportivo

Actividad Final de Minería de Datos. Un modelo **Random Forest** estima si un jugador tiene perfil **Élite** (Top 25 % del valor de mercado) a partir de su edad, estatura y posición, usando el dataset de Transfermarkt almacenado en MySQL. El modelo se despliega como **API REST con FastAPI** y se consume desde un aplicativo web.

## Flujo

```
MySQL (players) → exportar_modelo.py → models/modelo_scouting_rf.joblib → main_api.py (FastAPI) ⇄ index.html
```

## Estructura

| Archivo | Función |
| --- | --- |
| `src/database/connection2.py` | Conector a MySQL (SQLAlchemy + PyMySQL) |
| `exportar_modelo.py` | Preprocesamiento, diagnóstico de overfitting, GridSearchCV, ROC-AUC y exportación del modelo |
| `models/modelo_scouting_rf.joblib` | Modelo entrenado con sus columnas y medianas de imputación |
| `main_api.py` | API REST: `POST /predict` y `GET /` (sirve la página) |
| `index.html` | Aplicativo web "Ficha de Scouting" |
| `inferencia_nuevos_jugadores.py` | Prueba de predicción sin API |

## Ejecución

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Opcional: reentrenar (requiere MySQL con la tabla players y un archivo .env, ver .env.example)
python exportar_modelo.py

# Levantar la API y el aplicativo
uvicorn main_api:app
```

- Aplicativo web: http://127.0.0.1:8000
- Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

## Resultados

| Métrica (conjunto de prueba) | Valor |
| --- | --- |
| ROC-AUC Random Forest | 0.694 |
| ROC-AUC Árbol de decisión | 0.678 |
| Recall clase Élite | 0.77 |
| F1 clase Élite | 0.51 |

La edad es la variable más importante (76 %), seguida de la estatura (13 %) y la posición de portero (8 %).

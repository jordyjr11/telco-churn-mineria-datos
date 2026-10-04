# Predicción del abandono de clientes (Telco Churn)

Práctica experimental 1 · Minería de Datos · Universidad Estatal Amazónica  
Autor: Jordy Josué Parión Aigaje

Aplicación del proceso de minería de datos al conjunto *Telco Customer Churn* (Kaggle): exploración,
preprocesamiento, segmentación con K-Means, comparación de modelos de clasificación, ajuste de un
Random Forest, API y WebApp.

## Estructura

| Carpeta | Contenido |
|---|---|
| `notebook/` | Cuaderno `Telco_Churn_Mineria_Datos.ipynb` (Google Colab) |
| `data/` | Base de datos `WA_Fn-UseC_-Telco-Customer-Churn.csv` ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) |
| `modelo/` | `modelo_churn.pkl` (pipeline: preprocesamiento + Random Forest) y `umbral.json` |
| `app/` | Código fuente de la API (FastAPI, `app.py`) y de la WebApp (`index.html`). Desarrollados con asistencia de IA |
| `docs/` | Informe y cuaderno en PDF |

## Ejecutar la API y la WebApp

```bash
cd app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

- WebApp: http://127.0.0.1:8000/
- Documentación de la API (Swagger): http://127.0.0.1:8000/docs
- Endpoints: `GET /health`, `POST /predict`

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure":3,"Contract":"Month-to-month","InternetService":"Fiber optic","MonthlyCharges":85}'
```

## Nota
El modelo se entrenó con la versión de scikit-learn indicada en `app/requirements.txt`; usa esa misma versión.

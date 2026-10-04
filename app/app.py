"""API de predicción de abandono (Telco Churn) + WebApp.

Ejecutar desde la carpeta app/:  uvicorn app:app --reload
Web:       http://127.0.0.1:8000/
Swagger:   http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODELO_DIR = BASE.parent / "modelo"  # carpeta del repositorio con modelo_churn.pkl y umbral.json
if not MODELO_DIR.exists():
    MODELO_DIR = BASE
modelo = joblib.load(MODELO_DIR / "modelo_churn.pkl")  # pipeline completo (prep + Random Forest)

# El umbral óptimo se guarda desde el notebook (umbral.json). Si no existe, se usa 0.5.
try:
    UMBRAL = float(json.loads((MODELO_DIR / "umbral.json").read_text())["umbral"])
except Exception:
    UMBRAL = 0.5

SiNo = Literal["Yes", "No"]

app = FastAPI(title="API Telco Churn", version="1.0",
              description="Predice la probabilidad de abandono de un cliente de telecomunicaciones.")


class Cliente(BaseModel):
    gender: Literal["Female", "Male"] = "Female"
    SeniorCitizen: SiNo = "No"
    Partner: SiNo = "No"
    Dependents: SiNo = "No"
    tenure: int = Field(3, ge=0, le=72, description="Meses de antigüedad")
    PhoneService: SiNo = "Yes"
    MultipleLines: SiNo = "No"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    OnlineSecurity: SiNo = "No"
    OnlineBackup: SiNo = "No"
    DeviceProtection: SiNo = "No"
    TechSupport: SiNo = "No"
    StreamingTV: SiNo = "No"
    StreamingMovies: SiNo = "No"
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    PaperlessBilling: SiNo = "Yes"
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"] = "Electronic check"
    MonthlyCharges: float = Field(75.0, ge=0)
    TotalCharges: Optional[float] = Field(None, ge=0, description="Si se omite: tenure x MonthlyCharges")


def preparar(c: Cliente) -> pd.DataFrame:
    """Repite el feature engineering del notebook (Fase 3.2) sobre un cliente nuevo."""
    d = c.model_dump()
    if d["TotalCharges"] is None:
        d["TotalCharges"] = d["tenure"] * d["MonthlyCharges"]
    df = pd.DataFrame([d])

    servicios = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["n_servicios"] = (df[servicios] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["gasto_medio_mensual"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["variacion_cargo"] = df["MonthlyCharges"] - df["gasto_medio_mensual"]
    df["tenure_grupo"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    prot = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["tiene_proteccion"] = np.where((df[prot] == "Yes").any(axis=1), "Yes", "No")
    return df


@app.get("/health")
def health():
    return {"estado": "ok", "umbral": UMBRAL}


@app.post("/predict")
def predict(cliente: Cliente):
    proba = float(modelo.predict_proba(preparar(cliente))[0, 1])
    if proba >= UMBRAL:
        nivel = "Alto"
    elif proba >= UMBRAL / 2:
        nivel = "Medio"
    else:
        nivel = "Bajo"
    return {"probabilidad_abandono": round(proba, 4),
            "umbral": round(UMBRAL, 4),
            "prediccion": "Abandona" if proba >= UMBRAL else "Se queda",
            "nivel_riesgo": nivel}


@app.get("/", include_in_schema=False)
def web():
    return FileResponse(BASE / "index.html")

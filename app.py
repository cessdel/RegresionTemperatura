import json
import multiprocessing
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from modelo import ReggressionModel  # Asegúrate de importar tu modelo correctamente
import os
import glob

# Configuración del logger
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

model_handler = None  # Inicializamos como None para cargarlo después en lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manejo del ciclo de vida de la aplicación FastAPI. 
    Se carga el modelo más reciente al iniciar y se cierra cualquier recurso al apagar.
    """
    global model_handler
    logging.info("Server startup event triggered.")
    print("Server startup event triggered.")
    
    # Encuentra el archivo de modelo más reciente en el directorio
    model_files = glob.glob('modelRegression_*.pkl')
    if model_files:
        # Ordena los archivos por fecha de modificación (más reciente primero)
        latest_model_file = max(model_files, key=os.path.getmtime)
        model_handler = ReggressionModel(latest_model_file)
        logging.info(f"Model loaded successfully from {latest_model_file}.")
        print(f"Model loaded successfully from {latest_model_file}.")
    else:
        logging.error("No model files found.")
        print("No model files found.")
    
    yield  # Deja que la aplicación FastAPI funcione
    
    # Cierra cualquier recurso cuando la aplicación se apaga
    if model_handler:
        model_handler.close()
        logging.info("Server shutdown event triggered.")
        print("Server shutdown event triggered.")


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping():
    return {"message": "Servicio en línea. Modelo de regresión de temperatura."}

@app.get("/predict/short_term")
async def predict_short_term_temperatures(minutes: int = 1):
    """
    Endpoint para predecir las próximas X minutos de temperatura.
    Por defecto, predice para 1 minuto (corto plazo).
    """
    global model_handler
    try:
        # Verifica que el modelo esté cargado
        if model_handler is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")
        
        # Predicción de corto plazo basado en el número de minutos proporcionado
        results = model_handler.predict_short_term_temperatures(minutes=minutes)
        return {"future_temperatures": results}
    
    except Exception as e:
        logging.error(f"Error predicting short-term temperatures: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting short-term temperatures: {str(e)}")



@app.get("/predict/future")
async def predict_future_temperatures():
    """
    Endpoint para predecir las próximas 12 horas de temperatura.
    """
    global model_handler
    try:
        # Asegurarse de que el modelo esté cargado
        if model_handler is None:
            raise HTTPException(status_code=500, detail="Model is not loaded.")
        
        results = model_handler.predict_future_temperatures(hours=12)
        return {"future_temperatures": results}
    
    except Exception as e:
        logging.error(f"Error predicting future temperatures: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting future temperatures: {str(e)}")


@app.get("/status")
async def status():
    """
    Endpoint para obtener la información del último modelo recibido y su estado de entrenamiento.
    """
    global model_handler
    
    if model_handler is not None:
        # Obtén la fecha de modificación del archivo del modelo
        model_file = model_handler.model_path
        if os.path.exists(model_file):
            modification_time = os.path.getmtime(model_file)
            modification_date = datetime.fromtimestamp(modification_time).isoformat()
        else:
            modification_date = "Unknown"

        return {
            "status": "active",
            "model_file": model_file,
            "last_modified": modification_date
        }
    else:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

@app.post("/upload-model/")
async def upload_model(file: UploadFile = File(...)):
    """
    Endpoint para recibir un archivo .pkl del modelo, guardar una versión histórica,
    y cargar el nuevo modelo.
    """
    global model_handler
    try:
        # Crear un nombre de archivo con la fecha y hora actuales
        timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        historic_model_path = f"modelRegression_{timestamp}.pkl"

        # Guardar el archivo subido con un nombre de archivo único
        with open(historic_model_path, "wb") as f:
            f.write(await file.read())

        # Cargar el nuevo modelo
        model_handler = ReggressionModel(historic_model_path)
        logging.info(f"New model uploaded and loaded successfully. Stored as {historic_model_path}.")
        print(f"New model uploaded and loaded successfully. Stored as {historic_model_path}.")
        
        return {"status": f"Model uploaded and loaded successfully. Stored as {historic_model_path}."}
    
    except Exception as e:
        logging.error(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")
    
    
@app.post("/score/")
async def score(metrics: dict):
    """
    Endpoint para recibir y almacenar las métricas del modelo.
    """
    try:
        metrics_file = "model_metrics.json"
        with open(metrics_file, "a") as f:
            json.dump(metrics, f)
            f.write("\n")  # Escribe en una nueva línea para cada métrica

        logging.info(f"Received and saved metrics: {metrics}")
        return {"status": "Metrics received and saved successfully."}
    
    except Exception as e:
        logging.error(f"Error saving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving metrics: {str(e)}")
@app.get("/metrics/")
async def get_metrics():
    """
    Endpoint para obtener las métricas del modelo.
    """
    metrics_file = "model_metrics.json"
    try:
        metrics = []
        # Leer el archivo de métricas
        with open(metrics_file, "r") as f:
            for line in f:
                metrics.append(json.loads(line.strip()))

        logging.info(f"Metrics retrieved successfully: {metrics}")
        return {"metrics": metrics}

    except Exception as e:
        logging.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    multiprocessing.freeze_support()
    print("-----------------Servidor iniciado----------------")
    logging.info("Servidor iniciado")
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)

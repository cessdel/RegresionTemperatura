from influxdb_client import InfluxDBClient
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging




class ReggressionModel:
    
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.client = InfluxDBClient(
            url='http://192.168.0.100:8086',
            token='_VmYvh5wFRpx8EO7ntZpuEW-o0h3bWHnsBC_r5UkLi3hsZuHNkCTAhogWF-fkcpNRcT96SpDJw-Ek0cVa9ce6A==',
            org='SmartHome'
        )
        self.training_status = "Not started"

    def load_model(self):
        try:
            model = joblib.load(self.model_path)
            logging.info("Model loaded successfully.")
            return model
        except FileNotFoundError:
            model = LinearRegression()  # Modelo por defecto si no existe
            logging.error("Model not found, initialized a new LinearRegression model.")
            return model

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")


    def predict_future_temperatures(self, hours=12):
        now = datetime.now()  # Utiliza UTC
        print(f"Current time: {now}")
        next_hour = (now + timedelta(hours=1)).replace(minute=now.minute, second=0, microsecond=0)

        future_timestamps = [next_hour + timedelta(hours=i) for i in range(hours)]
        
        # Imprime los timestamps futuros
        for ft in future_timestamps:
            print(f"Future timestamp: {ft}")

        future_timestamps_seconds = np.array([dt.timestamp() for dt in future_timestamps]).reshape(-1, 1)
        print(f"Future timestamps in seconds: {future_timestamps_seconds}")

        predicted_temperatures = self.model.predict(future_timestamps_seconds)
        print(f"Predicted temperatures: {predicted_temperatures}")

        results = []
        for ts, temp in zip(future_timestamps_seconds, predicted_temperatures):
            predicted_time = pd.to_datetime(ts[0], unit='s', utc=True)  # Asegúrate de que esté en UTC
            predicted_time_local = predicted_time.tz_convert('America/Mexico_City')  # Ajusta la zona horaria local
            results.append({
                "timestamp": predicted_time_local.isoformat(),
                "predicted_temperature": temp
            })
        logging.info("Future temperatures solicited.")

        return results

    def evaluate_model(model, X_test, y_test):
        """
        Evaluar el modelo con un conjunto de datos de prueba y retornar métricas.
        """
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    
    def predict_short_term_temperatures(self, minutes=1):
        """
        Predice la temperatura para los próximos 'minutes' minutos.
        Se predice cada 10 segundos dentro del horizonte de tiempo especificado.
        """
        now = datetime.now()  # Utiliza UTC
        print(f"Current time: {now}")

        # Calcula los timestamps futuros en intervalos de 10 segundos
        future_timestamps = [now + timedelta(seconds=i * 10) for i in range((minutes * 60) // 10)]

        # Imprime los timestamps futuros
        for ft in future_timestamps:
            print(f"Future timestamp: {ft}")

        # Convierte los timestamps a segundos desde la época (Unix timestamp)
        future_timestamps_seconds = np.array([dt.timestamp() for dt in future_timestamps]).reshape(-1, 1)
        print(f"Future timestamps in seconds: {future_timestamps_seconds}")

        # Realiza la predicción de temperaturas
        predicted_temperatures = self.model.predict(future_timestamps_seconds)
        print(f"Predicted temperatures: {predicted_temperatures}")

        # Convertir los timestamps y predicciones a un formato legible
        results = []
        for ts, temp in zip(future_timestamps_seconds, predicted_temperatures):
            predicted_time = pd.to_datetime(ts[0], unit='s', utc=True)  # Asegúrate de que esté en UTC
            predicted_time_local = predicted_time.tz_convert('America/Mexico_City')  # Ajusta la zona horaria local
            results.append({
                "timestamp": predicted_time_local.isoformat(),
                "predicted_temperature": temp
            })
        logging.info("Short-term future temperatures requested.")

        return results



    def close(self):
        self.client.close()
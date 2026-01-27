# backend/app/repositories/firestore/metrics.py
from app.repositories.firestore.client import db
from datetime import datetime

METRICS_COLLECTION = "model_performance_metrics"

class MetricsLogger:
    """
    Logs algorithm performance metrics to Firestore
    """
    @staticmethod
    def log_algorithm_metrics(algorithm: str, execution_time: float, dataset_size: int, num_ranked_jobs: int):
        db.collection(METRICS_COLLECTION).add({
            "algorithm": algorithm,
            "execution_time": execution_time,
            "dataset_size": dataset_size,
            "num_ranked_jobs": num_ranked_jobs,
            "timestamp": datetime.utcnow()
        })

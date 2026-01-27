# backend/app/core/logging.py
import logging
import sys
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_no": record.lineno
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Configure logger
logger = logging.getLogger("resume_analyzer")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(JsonFormatter())
logger.addHandler(ch)

# Optional: helper functions
def log_info(msg: str):
    logger.info(msg)

def log_warning(msg: str):
    logger.warning(msg)

def log_error(msg: str):
    logger.error(msg)

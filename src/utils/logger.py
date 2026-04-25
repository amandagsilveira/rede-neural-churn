import logging
import json
import sys
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """
    Formatador customizado para forçar os Logs a saírem no formato JSON
    (Apropriado para ingestão em ElasticSearch, AWS CloudWatch, etc).
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        # Se houver dados métricos/extras atrelados ao aviso, engloba no JSON
        if hasattr(record, "metrics"):
            log_record["metrics"] = record.metrics
            
        return json.dumps(log_record, ensure_ascii=False)

def get_core_logger(name="ChurnLogger"):
    """
    Constrói e retorna um robô de registro estruturado para o sistema.
    """
    logger = logging.getLogger(name)
    
    # Previne que múltiplos outputs sejam criados se o código rodar várias vezes
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JsonFormatter())
        logger.addHandler(console_handler)
        
    return logger

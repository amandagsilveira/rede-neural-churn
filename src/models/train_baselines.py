import os
import sys

# Adiciona a raiz do projeto no caminho do Python para importarmos a 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc

from src.data.preprocess import load_and_split_data, build_preprocessor
from src.utils.logger import get_core_logger

logger = get_core_logger(__name__)

def evaluate_and_log(pipeline, model_name, X_test, y_test):
    """
    Simula predições e faz o tracking do modelo dentro do MLFlow.
    """
    # Predições
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] # Pega a propabilidade da classe Positiva (Churn=1)
    
    # Métricas Especiais do Canvas
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    # Gravando no MLFlow pra sempre!
    with mlflow.start_run(run_name=model_name):
        # Loga hyperparâmetros (que modelo que é)
        mlflow.log_param("model_type", model_name)
        
        # Loga nossa Prova Financeira
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("pr_auc", pr_auc)
        
        # Salva o pipeline matematico inteiro (Transformers + Algoritmo)
        mlflow.sklearn.log_model(pipeline, "model_pipeline")
        
        # LOGGING ESTRUTURADO AO INVÉS DO PRINT!
        logger.info(
            f"Registro do experimento {model_name} concluído.", 
            extra={'metrics': {'f1_score': round(f1, 4), 'pr_auc': round(pr_auc, 4)}}
        )

def main():
    logger.info("Iniciando rotina pesada de Treinamento de Baselines...")
    data_path = os.path.join("data", "raw", "Telco-Customer-Churn.csv")
    
    # 1. Carrega Dados usando nossa caixa preta da src.data
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    preprocessor = build_preprocessor(X_train)
    
    # 2. Configura e Inicializa MLFlow Local
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Baselines")
    
    # 3. Treinar e Registrar: DummyClassifier (O pior caso possivel)
    # Strategy 'prior': Ele sempre vai apostar na classe que mais aparece (No Churn)
    dummy_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DummyClassifier(strategy='prior'))
    ])
    dummy_pipeline.fit(X_train, y_train)
    evaluate_and_log(dummy_pipeline, "DummyClassifier-Prior", X_test, y_test)
    
    # 4. Treinar e Registrar: Regressão Logística (Baseline do Scikit)
    # class_weight='balanced': Dá peso maior pra classe Churn penalizando Falsos Negativos
    logreg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    logreg_pipeline.fit(X_train, y_train)
    evaluate_and_log(logreg_pipeline, "LogisticRegression-Balanced", X_test, y_test)
    
    logger.info("Baselines registrados com sucesso no MLFlow!")
    logger.info("Instrução visualização UI", extra={"metrics": {"comando": "mlflow ui --backend-store-uri sqlite:///mlflow.db"}})

if __name__ == "__main__":
    main()

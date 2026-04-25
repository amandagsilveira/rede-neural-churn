import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_and_split_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Carrega o dataset e faz a divisão entre features e variável alvo.
    """
    df = pd.read_csv(filepath)
    
    # Tratamento inicial 
    df = df.drop('customerID', axis=1) # ID não tem poder preditivo

    # Removendo Features baseadas na nossa decisão de arquitetura
    # 1. TotalCharges: Causa multicolinearidade pois é Tenure * MonthlyCharges
    # 2. gender: Removido por restrição ética do Model Card
    if 'TotalCharges' in df.columns:
        df = df.drop('TotalCharges', axis=1)
    if 'gender' in df.columns:
        df = df.drop('gender', axis=1)
    
    # Separar Target e as Features
    X = df.drop('Churn', axis=1)
    # Convertendo Churn de Yes/No para 1/0
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Split de Validação (Hold-out estratificado dado o desbalanceamento)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def build_preprocessor(X_train: pd.DataFrame):
    """
    Retorna uma pipeline unificada do Scikit-Learn que tratará qualquer 
    lixo de dados futuro na nossa AWS e fará as codificações (StandardScaler, etc).
    """
    # Identificando dinamicamente quais colunas são números e quais são textos
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Como processar sub-grupos de matemática
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Se falhar, bota mediana
        ('scaler', StandardScaler())                   # Redes neurais odeiam numeros altos
    ])

    # Como processar sub-grupos de texto (Gender, Contract...)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary')) # Vira colunas 0 ou 1
    ])

    # O "Liquidificador"
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    print("Módulo de preprocessamento inicializado.")

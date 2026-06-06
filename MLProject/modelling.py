import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.set_experiment("CI_CD_Fraud_Detection")
    
    df = pd.read_csv("clean_dataset.csv")
    X = df.drop('Fraud_Flag', axis=1).select_dtypes(include=['number']).astype(float)
    y = df['Fraud_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(rf, "model")
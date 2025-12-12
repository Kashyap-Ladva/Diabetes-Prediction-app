import pandas as pd 
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

data_path = Path("data/diabetes.csv")
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True) #models folder if not exist
scaler_path = model_dir / "scaler.pkl"
model_path = model_dir / "diabetes_model.pkl"


if not data_path.exists():
    raise FileNotFoundError(f"{data_path} not found. ")

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put diabetes.csv in data/ folder.")
    return pd.read_csv(path)

def train_and_save_model():
    df = load_data(data_path) #load data

    x = df.drop("Outcome" , axis = 1) #Features
    y = df["Outcome"] #Target

    X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42 , stratify=y) #training and testing split

    scaler = StandardScaler() #returns numpy arrays 
    X_train_scaled = scaler.fit_transform(X_train) #scaling
    X_test_scaled = scaler.transform(X_test) #avoid leakage

    model = LogisticRegression() #model
    model.fit(X_test_scaled,y_train)

    y_pred = model.predict(X_test_scaled) #evaluation

    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:") 
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(scaler, model_dir / "scaler.pkl") #saving scaler and model
    joblib.dump(model, model_dir / "diabetes_model.pkl")

if __name__ == "__main__":
    train_and_save_model()

    









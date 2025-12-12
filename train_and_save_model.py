# train_and_save_model.py (fixed, defensive)
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path("data/diabetes.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put diabetes.csv in data/ folder.")
    return pd.read_csv(path)

def train_and_save_model():
    df = load_data(DATA_PATH)
    if "Outcome" not in df.columns:
        raise ValueError("CSV must contain 'Outcome' column")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    print("Full dataset shapes:", "X:", X.shape, "y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sanity checks: sizes must match
    print("After split shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check scaled shapes
    print("After scaling shapes:")
    print("X_train_scaled:", X_train_scaled.shape, "X_test_scaled:", X_test_scaled.shape)

    # Defensive sanity: check lengths before fitting
    if X_train_scaled.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Mismatch: X_train has {X_train_scaled.shape[0]} rows but y_train has {y_train.shape[0]} rows."
            " Check variable names and splitting logic."
        )

    # Create and fit model using TRAIN data (not test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)   # <- correct: use X_train_scaled with y_train

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save scaler and model
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved scaler -> {SCALER_PATH} ({SCALER_PATH.stat().st_size} bytes)")
    print(f"Saved model  -> {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")

if __name__ == "__main__":
    train_and_save_model()

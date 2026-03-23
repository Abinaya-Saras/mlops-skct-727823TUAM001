print("TRAINING STARTED")

import mlflow
import mlflow.sklearn
import pandas as pd
import time
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

ROLL_NO = "727823TUAM001"
DATASET_NAME = "Diabetes"

mlflow.set_experiment(f"SKCT_{ROLL_NO}_{DATASET_NAME}")

# Load dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Models
models = [
    LinearRegression(),
    RandomForestRegressor(),
]

best_r2 = -1
best_model = None

for i in range(12):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

    model = models[i % len(models)]

    if isinstance(model, RandomForestRegressor):
        model.set_params(
            n_estimators=300,
            max_depth=10,
            random_state=i
        )

    start = time.time()

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAPE", mape)

        # Log params
        mlflow.log_param("model", type(model).__name__)
        mlflow.log_param("random_seed", i)

        # Tags
        mlflow.set_tag("student_name", "Abinaya Saras")
        mlflow.set_tag("roll_number", ROLL_NO)
        mlflow.set_tag("dataset", DATASET_NAME)

        # Operational metrics
        mlflow.log_metric("training_time_seconds", time.time() - start)
        mlflow.log_metric("model_size_mb", len(str(model)) / 1e6)
        mlflow.log_metric("n_features", X.shape[1])

        # Save model
        mlflow.sklearn.log_model(model, name="model")

        print(f"Run {i} → {type(model).__name__}, R2: {r2:.4f}")

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

with mlflow.start_run(run_name="BEST_MODEL"):
    mlflow.log_metric("best_R2", best_r2)
    mlflow.sklearn.log_model(best_model, name="best_model")

print("BEST R2:", best_r2)

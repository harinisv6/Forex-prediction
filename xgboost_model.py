from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.data_loader import load_tabular_data
import numpy as np

def train_xgboost():
    X, y = load_tabular_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("XGBoost MSE:", mse)

    return model, X_test, y_test, y_pred

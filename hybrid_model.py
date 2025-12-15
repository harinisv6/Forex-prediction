from xgboost_model import train_xgboost
from tft_model import train_tft
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Train models and get predictions
xgb_model, X_test, y_test, xgb_pred = train_xgboost()
tft_model, tft_pred = train_tft()

# Align predictions length
min_len = min(len(xgb_pred), len(tft_pred))
xgb_pred = xgb_pred[:min_len]
tft_pred = tft_pred[:min_len]
y_test = y_test[:min_len]

# Hybrid: Weighted average
final_pred = 0.5*xgb_pred + 0.5*tft_pred
print("Hybrid Model MSE:", mean_squared_error(y_test, final_pred))

# Meta-Model (Stacking)
meta_X = np.vstack([xgb_pred, tft_pred]).T
meta_model = LinearRegression()
meta_model.fit(meta_X, y_test)
final_meta_pred = meta_model.predict(meta_X)

print("Meta-Model MSE:", mean_squared_error(y_test, final_meta_pred))

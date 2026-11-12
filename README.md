# TSAF-Ex-8-Modeling-and-Forecasting-with-ARMA-Processes

## AIM:
To model and forecast a time series using an ARMA(p,q) process

## PROGRAM:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(99)

ar = np.array([1])

ma = np.array([1, 0.6, 0.3])

arma_process = ArmaProcess(ar, ma)
data = arma_process.generate_sample(nsample=200)
series = pd.Series(data)

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Simulated MA(2) Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, ax=ax[1])
plt.show()

train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]
print("Training observations:", len(train))
print("Testing observations:", len(test))
model_ma1 = ARIMA(train, order=(0, 0, 1))
fit_ma1 = model_ma1.fit()
model_ma2 = ARIMA(train, order=(0, 0, 2))
fit_ma2 = model_ma2.fit()
model_arma12 = ARIMA(train, order=(1, 0, 2))
fit_arma12 = model_arma12.fit()

print("\n--- Model Information Criteria Comparison ---")
print(f"MA(1) | AIC: {fit_ma1.aic:.2f} | BIC: {fit_ma1.bic:.2f}")
print(f"MA(2) | AIC: {fit_ma2.aic:.2f} | BIC: {fit_ma2.bic:.2f}")
print(f"ARMA(1,2) | AIC: {fit_arma12.aic:.2f} | BIC: {fit_arma12.bic:.2f}")
forecast = fit_ma2.forecast(steps=len(test))

mae = mean_absolute_error(test, forecast)

mse = mean_squared_error(test, forecast)

rmse = np.sqrt(mse)

print("\n--- Forecast Error Metrics for MA(2) ---")
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))

plt.figure(figsize=(12, 5))
plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Test Data")
plt.plot(test.index, forecast, label="MA(2) Forecast", color="red")
plt.title("MA(2) Forecast vs Actual Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

```

## OUTPUT:

<img width="473" height="538" alt="image" src="https://github.com/user-attachments/assets/e15509c8-09ac-403c-b63f-21c331205a8a" />
<img width="456" height="223" alt="image" src="https://github.com/user-attachments/assets/bc7c142f-84cf-442a-b992-0bac3d65f8bf" />


<img width="1032" height="692" alt="image" src="https://github.com/user-attachments/assets/64bca88b-9e51-4eea-a0cc-182089c593c2" />

<img width="816" height="375" alt="image" src="https://github.com/user-attachments/assets/c05a52cb-9fb7-4598-9a48-22cf39326087" />

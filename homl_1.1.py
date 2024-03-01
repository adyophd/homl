# Training and running a linear model in Scikit-Learn

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True) # turns off scientific notation
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats

# Download and prepare the data
lifesat = pd.read_csv("https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv")

GDP = lifesat[["GDP per capita (USD)"]].values
Satis = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500,62_500,4,9])
# plt.show()
# Suppressing plot for now

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(GDP,Satis)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Since we're using the entire dataset for training, this R-squared is based on the same data.
# For a more accurate assessment, consider splitting your data into training and test sets.
print("R-squared:", model.score(GDP, Satis))

# Make a prediction for Cyprus
GDP_Cyprus = [[37_655.2]] # Cyprus GDP per capita in 2020
cyprus_prediction = model.predict(GDP_Cyprus)
print("Predicted life satisfaction for Cyprus (GDP per capita of 37,655.2 USD):", cyprus_prediction)
# Output: 6.3016

# Optional: Calculate Mean Squared Error (MSE) if you have a test set or want to calculate it on the training set
mse = mean_squared_error(Satis, model.predict(GDP))
print("Mean Squared Error:", mse)

# y_hat = 0.00006779(GDP) + 3.74904943
Satis_hat = (0.00006779*GDP) + 3.74904943

# R^2 = 0.7272610933272652
# MSE = 0.15394596065527696

# Calculate MSE manually and compare to the prebuilt mse output



# print("Satis_hat:", Satis_hat)
# print(type(Satis_hat))
# print("Satis:", Satis)
# print(type(Satis))

def calculate_mse(y_hat, y_actual):
    n = len(y_actual)
    squared_differences_from_prediction = []
    squared_differences_from_mean = []

    for y_hat_i, y_actual_i in zip(y_hat, y_actual):
        squared_differences_from_prediction.append((y_actual_i - y_hat_i) ** 2)
        squared_differences_from_mean.append((y_actual_i - np.mean(Satis)) ** 2)

    manual_SS_residual = sum(squared_differences_from_prediction)
    manual_SS_total = sum(squared_differences_from_mean)

    manual_MSE = manual_SS_residual / n

    return manual_SS_residual, manual_SS_total, manual_MSE

my_ss_residual, my_ss_total, my_mse = calculate_mse(Satis_hat,Satis)

print("Manual SS Residual:", my_ss_residual)
print("Manual SS Total:", my_ss_total)
print("Manual R^2:", (1-(my_ss_residual/my_ss_total)))
print("Manual MSE:", my_mse)
print("RMSE:", np.sqrt(my_mse))
# RMSE is the average prediction error (i.e., Satis_i = Satis_hat_i +/- 0.39)
# Better models will have smaller RMSE

# Next is to calculate the F statistic and p-value of the model

# Compute statsig test
GDP_with_intercept = sm.add_constant(GDP) # Unlike scikit learn's LinearRegression, OLS doesn't automatically add an intercept
model_test_results = sm.OLS(Satis, GDP_with_intercept).fit()
print(model_test_results.summary())
# F(1,25) = 66.66, p <.001, R^2 = 0.727


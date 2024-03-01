# Convert 1.1 to k_regression

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True) # turns off scientific notation
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
model = KNeighborsRegressor(n_neighbors=3)

# Train the model
model.fit(GDP,Satis)

GDP_Cyprus = [[37_655.2]]
cyprus_prediction = model.predict(GDP_Cyprus)
print("Predicted life satisfaction for Cyprus (GDP per capita of 37,655.2 USD):", cyprus_prediction)
# Output: 6.3333

# What this model (k-nearest neighbors regression) is doing is finding the y values for the closest n number of neighbors
# and averaging them. This might be useful if there's ever a poor linear model (so using regression equation is out)
# but you want a best guess at the value of another item.
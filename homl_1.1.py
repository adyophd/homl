# Training and running a linear model in Scikit-Learn

# Install packages in Pycharm > Settings > Project > Interpreter
# Ideally, automate this

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare the data
# Can I grab the data from the original source rather than move it to my own?
data_root = "https://github.com/ageron/data/raw/main/"

lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500,62_500,4,9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(x,y)

# Make a prediction for Cyprus
X_new = [[37_655.2]] # Cyprus GDP per capita in 2020
print(model.predict(X_new)) # output: [[6.3016567]]
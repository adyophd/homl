from pathlib import Path # not clear on this function
import pandas as pd
import tarfile # not clear on tarballs
import urllib.request
import matplotlib.pyplot as plt

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball: # unclear on use of "with"
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# --------------------------
# Do some data exploration
# --------------------------

# pd.set_option('display.max_columns', None)
# housing.head()
# housing.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 20640 entries, 0 to 20639 <---- reveals 20640 rows of data (districts)
# Data columns (total 10 columns):
#  #   Column              Non-Null Count  Dtype
# ---  ------              --------------  -----
#  0   longitude           20640 non-null  float64
#  1   latitude            20640 non-null  float64
#  2   housing_median_age  20640 non-null  float64
#  3   total_rooms         20640 non-null  float64
#  4   total_bedrooms      20433 non-null  float64  <---- incomplete data
#  5   population          20640 non-null  float64
#  6   households          20640 non-null  float64
#  7   median_income       20640 non-null  float64  <---- income(1) = $10,000, cap at 15
#  8   median_house_value  20640 non-null  float64
#  9   ocean_proximity     20640 non-null  object

# housing["ocean_proximity"].value_counts()   <--- for categorical variables
# housing.describe() <---- for numerical variables
# housing.hist(bins = 50, figsize = (12,8)) <--- reveals caps on certain variables that be problematic for modeling




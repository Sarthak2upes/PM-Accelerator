# Weather Trend Forecasting using Machine Learning

## PM Accelerator Mission

Empowering professionals through AI, Product Management, and Technology innovation while creating real-world impact.

---

# Project Overview

This project analyzes the Global Weather Repository dataset and develops machine learning models to forecast weather trends by predicting temperature using various atmospheric and environmental factors.

The project demonstrates the complete data science workflow, including:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Forecasting Model Development
* Ensemble Learning
* Feature Importance Analysis
* Air Quality Analysis
* Climate Analysis
* Spatial Analysis

Dataset Source:
https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository

---

# Dataset Description

The dataset contains weather observations collected from locations around the world and includes:

* Temperature
* Wind Speed and Direction
* Humidity
* Pressure
* Cloud Cover
* Visibility
* UV Index
* Air Quality Indicators
* Latitude and Longitude
* Astronomical Features

The target variable used for forecasting is:

```text
temperature_celsius
```

---

# Data Preprocessing

## Missing Values

The dataset was checked for missing values using:

```python
df.isnull().sum()
```

No significant missing values were found in the selected features used for modeling, therefore no missing value imputation was required.

---

## Duplicate Records

No significant duplicate values were found in the selected features used for modeling, therefore no duplicate value imputation was required.


---

## Outlier Handling

Traditional IQR-based outlier removal was not used.

Weather datasets often contain genuine extreme observations such as:

* Heatwaves
* Storms
* Heavy rainfall
* High humidity events

Removing these values using statistical techniques like IQR could remove meaningful weather information.

Instead, domain-based thresholds were applied using realistic meteorological limits such as:

* Temperature: -60°C to 60°C
* Humidity: 0% to 100%
* Cloud Cover: 0% to 100%
* Pressure: 870 mb to 1085 mb
* UV Index: 0 to 15

This approach preserves valid weather extremes while filtering physically impossible values.

---

## Feature Selection

Several features were removed before model training.

Removed features include:

* temperature_fahrenheit
* feels_like_celsius
* feels_like_fahrenheit
* last_updated_epoch
* sunrise
* sunset
* moonrise
* moonset

### Reason

Some of these variables are directly derived from temperature.

Including them would introduce target leakage and artificially improve model performance.

The objective was to predict temperature from independent weather conditions rather than from variables calculated using temperature itself.

---

## Feature Scaling

Min-Max Normalization was applied to scale all features between 0 and 1.

```python
from sklearn.preprocessing import MinMaxScaler
```

Normalization helps ensure consistency across features measured on different scales.

---

# Exploratory Data Analysis

The following analyses were performed:

### Temperature Distribution

Analyzed the distribution and spread of temperature values across the dataset.

### Precipitation Distribution

Examined rainfall patterns and precipitation variability.

### Correlation Heatmap

Studied relationships between weather variables.

### Time-Based Analysis

Used the `last_updated` feature to analyze weather trends over time.

---

# Climate Analysis

Average temperatures were calculated and compared across different countries.

This analysis helped identify regions with higher and lower average temperatures and provided insight into climate variation across geographical locations.

---

# Air Quality Analysis

Air quality indicators were analyzed against weather conditions.

Examples include:

* PM2.5 vs Temperature
* PM10 vs Temperature

This helped explore environmental relationships between pollution levels and weather variables.

---

# Spatial Analysis

Geographical weather patterns were visualized using:

* Latitude
* Longitude
* Temperature

Scatter plots were used to observe how temperature varies across different regions of the world.

---

# Forecasting Models

Three forecasting models were developed and compared.

---

## Ridge Regression

Initially, Linear Regression was tested as a baseline model.

However, the dataset contains several highly correlated variables, including:

* wind_mph and wind_kph
* pressure_mb and pressure_in
* precip_mm and precip_in
* gust_mph and gust_kph

Additionally, one-hot encoding created many additional features.

These conditions introduced multicollinearity, causing Linear Regression to become unstable and produce extremely poor predictions with highly negative R² values.

To address this issue, Ridge Regression was used.

### Why Ridge Regression?

Ridge Regression applies L2 Regularization, which:

* Reduces coefficient instability
* Handles multicollinearity
* Improves generalization
* Produces more stable predictions

---

## Random Forest Regressor

Random Forest was selected because weather systems often exhibit complex non-linear relationships.

Advantages:

* Captures non-linear patterns
* Robust to outliers
* Handles feature interactions effectively
* Provides feature importance scores

Configuration:

```python
RandomForestRegressor(
    n_estimators=25,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
```

---

## Ensemble Model

A Voting Regressor was built using:

* Ridge Regression
* Random Forest Regressor

The purpose of the ensemble model was to combine the strengths of both approaches and improve forecasting performance.

---

# Model Performance

| Model                     | MAE  | RMSE | R² Score |
| ------------------------- | ---- | ---- | -------- |
| Ridge Regression          | 3.21 | 4.33 | 0.794    |
| Random Forest Regressor   | 2.80 | 3.87 | 0.835    |
| Voting Ensemble Regressor | 2.67 | 3.66 | 0.853    |

---

# Feature Importance

Feature importance analysis was performed using the Random Forest model.

This helped identify the most influential variables affecting temperature prediction.

Examples include:

* Humidity
* Pressure
* Wind Characteristics
* Cloud Cover
* Air Quality Indicators

---

# Results

Key findings from the project:

* Domain-based filtering preserved meaningful weather extremes.
* Ridge Regression successfully addressed multicollinearity issues.
* Random Forest captured non-linear weather relationships effectively.
* The ensemble model achieved the best forecasting performance.
* Air quality and spatial analyses provided additional environmental insights.
* The final ensemble model achieved an R² score of 0.853.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* KaggleHub

---

# Repository Structure

```text
├── Weather_Forecasting_Assessment.ipynb
├── README.md
├── requirements.txt
└── images/
```

---

# Author

**Sarthak Kothiyal**
B.Tech Computer Science Engineering (AI & ML)
UPES Dehradun

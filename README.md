# Weather Trend Forecasting using Machine Learning

## PM Accelerator Mission

Empowering professionals through AI, Product Management, and Technology innovation while creating real-world impact.

---

# Project Overview

This project analyzes the **Global Weather Repository Dataset** and builds machine learning models to forecast weather trends, specifically temperature prediction.

The project covers the complete data science workflow including:

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

The dataset contains weather information collected from locations around the world and includes:

* Temperature
* Wind Speed
* Humidity
* Pressure
* Cloud Cover
* Visibility
* UV Index
* Air Quality Metrics
* Geographical Coordinates
* Astronomical Information

The dataset contains approximately 149,000 weather observations and over 40 original features.

---

# Data Cleaning and Preprocessing

## Missing Values

Missing numerical values were handled using median imputation.

```python
df.fillna(df.median(numeric_only=True), inplace=True)
```

Median imputation was selected because it is less sensitive to extreme weather observations than mean imputation.

---

## Duplicate Records

Duplicate observations were removed.

```python
df.drop_duplicates(inplace=True)
```

---

## Outlier Handling

Traditional IQR-based outlier removal was intentionally not used.

Weather datasets often contain legitimate extreme observations such as:

* Heatwaves
* Storms
* High humidity events
* Extreme precipitation events

Removing such observations using IQR would remove meaningful climatic information.

Instead, domain-specific weather thresholds were applied.

Examples:

* Temperature between -60°C and 60°C
* Humidity between 0% and 100%
* Cloud Cover between 0% and 100%
* Pressure between 870 mb and 1085 mb
* UV Index between 0 and 15

This approach preserves genuine weather extremes while removing physically impossible values.

---

## Feature Selection

Several columns were removed before modeling.

Removed features included:

* temperature_fahrenheit
* feels_like_celsius
* feels_like_fahrenheit
* last_updated_epoch
* sunrise
* sunset
* moonrise
* moonset

### Why were these removed?

Some features were direct transformations of temperature.

Examples:

* temperature_fahrenheit
* feels_like_celsius
* feels_like_fahrenheit

Including them would introduce target leakage because the model could indirectly access the value it was trying to predict.

The objective was to create a model that predicts temperature from independent weather conditions.

---

## Feature Scaling

Min-Max Normalization was applied.

```python
MinMaxScaler()
```

This scales all features to a common range between 0 and 1.

---

# Exploratory Data Analysis

The following analyses were performed:

### Temperature Distribution

Understanding the spread of temperatures across global locations.

### Precipitation Distribution

Analyzing rainfall patterns.

### Correlation Heatmap

Studying relationships between numerical weather variables.

### Time-Based Temperature Trends

Analyzing temperature variation using the timestamp information.

---

# Climate Analysis

Average temperatures were calculated for different countries.

The hottest countries in the dataset were identified and visualized.

This provides insight into long-term climate differences between regions.

---

# Air Quality Analysis

Relationships between temperature and air quality indicators were explored.

Example:

* PM2.5 vs Temperature

This helps investigate possible environmental relationships between pollution and weather conditions.

---

# Spatial Analysis

Geographical weather patterns were visualized using:

* Latitude
* Longitude
* Temperature

Scatter plots were used to observe how temperature varies across different geographical regions.

---

# Forecasting Models

Three forecasting models were developed and compared.

---

## Ridge Regression

Ridge Regression was selected instead of standard Linear Regression.

### Why not Linear Regression?

The dataset contains many highly correlated variables.

Examples include:

* wind_mph and wind_kph
* pressure_mb and pressure_in
* precip_mm and precip_in
* gust_mph and gust_kph

After one-hot encoding categorical variables, the dataset also contained hundreds of additional features.

These conditions create multicollinearity.

Multicollinearity causes ordinary Linear Regression to become unstable and produce extremely poor predictions.

During experimentation, Linear Regression produced very large errors and negative R² values.

### Why Ridge Regression?

Ridge Regression introduces L2 Regularization, which helps:

* Reduce coefficient instability
* Handle multicollinearity
* Improve generalization
* Prevent excessively large coefficients

As a result, Ridge Regression produced significantly more stable predictions.

---

## Random Forest Regressor

Random Forest was used to capture complex non-linear relationships in weather data.

Configuration:

```python
RandomForestRegressor(
    n_estimators=25,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
```

Advantages:

* Handles non-linear relationships
* Robust to outliers
* Less affected by multicollinearity
* Provides feature importance scores

---

## Ensemble Model

A Voting Regressor was created using:

* Ridge Regression
* Random Forest Regressor

The objective was to combine the strengths of both models.

```python
VotingRegressor(
    [
        ("ridge", ridge),
        ("rf", rf)
    ]
)
```

---

# Model Performance

| Model            | MAE  | RMSE | R² Score |
| ---------------- | ---- | ---- | -------- |
| Ridge Regression | 3.21 | 4.33 | 0.794    |
| Random Forest    | 2.80 | 3.87 | 0.835    |
| Ensemble Model   | 2.67 | 3.66 | 0.853    |

---

# Feature Importance

Feature importance was extracted from the Random Forest model.

This helped identify the weather variables that contribute most to temperature prediction.

Examples of influential features include:

* Humidity
* Pressure
* Wind Characteristics
* Air Quality Indicators
* Cloud Cover

---

# Results and Insights

* Domain-based cleaning preserved meaningful weather extremes.
* Ridge Regression outperformed standard Linear Regression due to multicollinearity.
* Random Forest effectively captured non-linear weather relationships.
* The Ensemble Model achieved the best performance.
* Air quality and geographical analyses provided additional environmental insights.
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

# Future Improvements

* Time-series forecasting using dedicated forecasting models.
* Hyperparameter optimization.
* Gradient Boosting and XGBoost models.
* More advanced climate trend analysis.
* Interactive dashboards using Streamlit or Power BI.

---

# Author

Sarthak Kothiyal

B.Tech Computer Science Engineering (AI & ML)

UPES Dehradun

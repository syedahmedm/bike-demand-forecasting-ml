# Bike-Sharing Demand Forecasting with Machine Learning

Predicting hourly bike rental demand from weather and time features, comparing six regression models with 10-fold cross-validation and a tuned XGBoost final model.

**Tech:** Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter

## Business problem
Bike-sharing operators struggle with fluctuating demand: too few bikes at peak hours hurts service, too many off-peak wastes resources. This project forecasts hourly rentals so that bike redistribution, staffing and maintenance can be planned in advance.

## Dataset
- 8,760 hourly observations (one year), 14 raw columns
- Target: `Rented Bike Count`
- Predictors: hour, season, holiday, functioning day, temperature, humidity, wind speed, visibility, dew point, solar radiation, rainfall, snowfall
- Source: [add dataset name and link, e.g. UCI / Kaggle page]. Check its licence before including the CSV in this repo.

## Approach
1. **Data cleaning:** fixed data types (unit strings in temperature), dropped constant and >50% missing columns, standardized column names and dates.
2. **Missing values:** grouped median imputation for solar radiation (by hour and season); date-based mode imputation for seasons.
3. **Outliers:** percentile clipping (99th, then 98th) checked with skewness and kurtosis.
4. **Feature engineering:** cyclical `Hour_sin` / `Hour_cos` encoding, month, day of week and weekend flag, one-hot and binary encoding, MinMax scaling.
5. **Modeling:** Linear Regression, Decision Tree, Random Forest, KNN, SVM and XGBoost, evaluated with 10-fold CV and a 25% hold-out test set.
6. **Tuning:** `RandomizedSearchCV` (50 iterations, 10-fold) for XGBoost, plus feature importance.

## Results (hold-out test set)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 317.60 | 414.39 | 0.567 |
| Decision Tree | 128.72 | 234.50 | 0.861 |
| Random Forest | 98.12 | 168.52 | 0.928 |
| K-Nearest Neighbors | 142.64 | 231.29 | 0.865 |
| SVM | 379.54 | 543.36 | 0.256 |
| **XGBoost** | **99.80** | **163.37** | **0.933** |

After tuning, XGBoost scored MAE 102.05, RMSE 162.73 and R² 0.93 on the same test set, so the default and tuned models perform almost identically.

**Key insights:** demand is bimodal with commuter peaks at 08:00 and 18:00; winter demand is about 78% below summer; rentals rise with temperature up to about 30°C, a non-linear pattern that tree-based models capture and linear regression cannot.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook bike_demand_forecasting.ipynb
```
Place `BikeData.csv` in the same folder as the notebook (or in `data/` and update the path).

## My contribution
This was a group project for [course name, university]. My part: [e.g. data cleaning and feature engineering, model comparison, visualizations, report writing].

Team: [teammate names / GitHub links]

## Author
Syed Ahmed Musthak · [linkedin.com/in/syedahmedm](https://linkedin.com/in/syedahmedm)

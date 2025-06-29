# 🚗 US Accident Severity Prediction

This project aims to build a robust machine learning model to predict the **severity level of road accidents** in the US using features like weather conditions, time, and location data.

> 📊 **Goal**: Help traffic management systems and emergency services take proactive actions using accident severity forecasts.

---

## 📁 Dataset

- **Source**: [US Accidents - Kaggle Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/code)
- **Size**: ~3 million records, 47 columns
- **Features**: Geospatial (lat/lng), weather, time, infrastructure, road condition, etc.
- **Target**: `Severity` (1: Low Impact → 4: High Impact)

---

## ✅ Objectives

- Clean and preprocess the raw data.
- Engineer meaningful features.
- Handle class imbalance using SMOTE.
- Select top predictive features.
- Train and compare classification models.
- Evaluate and visualize performance.

---

## 🛠️ Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Jupyter Notebook
- SMOTE for class balancing
- Random Forest, Decision Tree, Logistic Regression
- LightGBM (optional)

---

## 🔍 Exploratory Data Analysis (EDA)

- **Missing Values**: Dropped or imputed based on strategy (e.g. calm wind speed = 0).
- **Target Class Imbalance**: Severity 2 dominates (77%).
- **Time Features**: Extracted `hour`, `weekday`, and `month` from `Start_Time`.
- **Top Cities with Accidents**:
  - Houston, Miami, LA were most frequent.
- **Weather Analysis**:
  - Accidents mostly occurred under `Fair` and `Clear` conditions.
- **Outliers**:
  - Removed using IQR method for numerical features.

---

## 🧠 Feature Engineering

- Created:
  - `Duration_Min` (accident duration)
  - `Is_Night`, `Is_Weekend` (from timestamp)
  - Text-based features: `Description_Length`, `Street_Length`
- Encoded:
  - Categorical features using `LabelEncoder`
- Selected:
  - Top features using `SelectFromModel` (Random Forest)

---

## ⚖️ Class Imbalance Handling

Used **SMOTE** to oversample minority severity classes:
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_selected, y_train)
````

---

## 🤖 Model Training

| Model               | Accuracy | F1-Score |
| ------------------- | -------- | -------- |
| ✅ Random Forest     | 0.85     | **0.85** |
| Decision Tree       | 0.79     | 0.79     |
| Logistic Regression | 0.42     | 0.51     |

---

## 📊 Visualizations

* 📈 Hourly, daily, and monthly accident distribution
* 📍 Top 10 accident-prone cities
* 🌦️ Most common weather conditions
* 🔥 Correlation heatmap between features
* 🎯 Feature importance plot (Random Forest)
* 🧾 Confusion Matrix of best model

---

## 🧠 Key Insights

* Accidents mostly happen during morning rush hours (7-9 AM).
* Most frequent in `Fair` or `Clear` weather—indicating human or infrastructure factors.
* Cities with high traffic volume show more accidents (e.g. Houston, Miami).
* Class imbalance was handled successfully, improving prediction for rare severity levels.

---

## 🧪 Next Steps

* 📈 Hyperparameter tuning using `GridSearchCV`
* 🌐 Deployment using Streamlit / Flask
* 🧠 Add NLP/Sentiment analysis on `Description` column
* 🗺️ Visualize accident hotspots using geospatial clustering (optional)

---

## 💾 Model Export

```python
import joblib
joblib.dump(best_model, "random_forest_us_accidents.pkl")
```

---


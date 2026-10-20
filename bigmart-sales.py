import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("train.csv")

df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].median())
df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Size"].mode()[0])

df["Item_Visibility"] = df["Item_Visibility"].replace(0, np.nan)
df["Item_Visibility"] = df["Item_Visibility"].fillna(df["Item_Visibility"].mean())
-
# Extract item category from Item_Identifier (first 2 characters)
df["Item_type_category"] = df["Item_Identifier"].apply(lambda x: x[0:2])
df = df.drop("Item_Identifier", axis=1)


df["outlet_age"] = 2026 - df["Outlet_Establishment_Year"]
df = df.drop("Outlet_Establishment_Year", axis=1)

df["Price_Per_Weight"] = df["Item_MRP"] / df["Item_Weight"]
item_mrp_mean = df.groupby("Item_Type")["Item_MRP"].mean()
df["Item_Type_Mean_MRP"] = df["Item_Type"].map(item_mrp_mean)

outlet_type_mean = df.groupby("Outlet_Type")["Item_Outlet_Sales"].mean()
df["Outlet_Type_Mean_Sales"] = df["Outlet_Type"].map(outlet_type_mean)

df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
    {"LF ": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Item_type_category"] = le.fit_transform(df["Item_type_category"])
df["Outlet_Identifier"] = le.fit_transform(df["Outlet_Identifier"])

df = pd.get_dummies(
    data=df,
    columns=[
        "Item_Fat_Content",
        "Item_Type",
        "Outlet_Size",
        "Outlet_Location_Type",
        "Outlet_Type",
    ],
    dtype="int",
)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop(columns="Item_Outlet_Sales", axis=1)
y = df["Item_Outlet_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler1 = StandardScaler()
x_train_scaled = scaler1.fit_transform(X_train)
x_test_scaled = scaler1.transform(X_test)


from sklearn.metrics import r2_score


from sklearn.linear_model import LinearRegression

le_model = LinearRegression()
le_model.fit(x_train_scaled, y_train)
y_pred = le_model.predict(x_test_scaled)
r2 = r2_score(y_test, y_pred)
print("Linear Regression R2 Score:", r2)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
)
rf_model.fit(x_train_scaled, y_train)
y_pred_rf = rf_model.predict(x_test_scaled)
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))


from sklearn.model_selection import RandomizedSearchCV

classifier_rf = RandomizedSearchCV(
    rf_model,
    {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    n_iter=20,
    cv=5,
    return_train_score=False,
)
classifier_rf.fit(x_train_scaled, y_train)

best_rf_model = classifier_rf.best_estimator_
y_pred_tuned_rf = best_rf_model.predict(x_test_scaled)
print("Tuned Random Forest R² Score:", r2_score(y_test, y_pred_tuned_rf))

#  XGBoost Regressor
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)
xgb_model.fit(x_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(x_test_scaled)
print("XGBoost R2 Score:", r2_score(y_test, y_pred_xgb))

# Voting Regressor
from sklearn.ensemble import VotingRegressor

ensemble_model = VotingRegressor(
    estimators=[("rf", rf_model), ("xgb", xgb_model)]
)
ensemble_model.fit(x_train_scaled, y_train)
y_pred_ensemble = ensemble_model.predict(x_test_scaled)
print("Ensemble Model R² Score:", r2_score(y_test, y_pred_ensemble))

import joblib

joblib.dump(rf_model, "best_rf_model.pkl")
joblib.dump(scaler1, "scaler1.pkl")
joblib.dump(X.columns.tolist(), "columns1.pkl")

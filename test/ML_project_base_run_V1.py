import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = r"C:\Users\eitanb\Documents\DS\ML\ML_project\DATA\Train.csv"
df = pd.read_csv(file_path)

# Convert saledate to datetime and extract year, month, and day
if 'saledate' in df.columns:
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
    df['sale_year'] = df['saledate'].dt.year
    df['sale_month'] = df['saledate'].dt.month
    df['sale_day'] = df['saledate'].dt.day
    df.drop(columns=['saledate'], inplace=True)

# Drop irrelevant columns
columns_to_drop = ["SalesID", "Unnamed: 0"]  # Drop IDs and redundant columns
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Drop columns with more than 80% missing values
missing_threshold = 0.8
missing_ratio = df.isnull().sum() / len(df)
columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
#df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Handle missing values
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)  # Impute numeric values with median

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)  # Impute categorical with mode or 'Unknown'

# Handle outliers
# Apply log transformation to SalePrice
df['SalePrice'] = np.log1p(df['SalePrice'])

# Cap MachineHoursCurrentMeter at the 99th percentile
if 'MachineHoursCurrentMeter' in df.columns:
    upper_limit = df['MachineHoursCurrentMeter'].quantile(0.99)
    df['MachineHoursCurrentMeter'] = np.where(df['MachineHoursCurrentMeter'] > upper_limit, upper_limit, df['MachineHoursCurrentMeter'])

# Fix YearMade errors
# Replace values below 1900 with NaN and impute with median
df.loc[df['YearMade'] < 1900, 'YearMade'] = np.nan
df['YearMade'].fillna(df['YearMade'].median(), inplace=True)

# Bucketize YearMade
bins = [1900, 1950, 1980, 2000, 2010, 2025]
labels = ['Pre-1950', '1950-1980', '1980-2000', '2000-2010', 'Post-2010']
df['YearMade_Bucket'] = pd.cut(df['YearMade'], bins=bins, labels=labels)

# Convert categorical variables to numeric using One-Hot Encoding or Label Encoding
label_encoders = {}
df = pd.get_dummies(df, columns=['YearMade_Bucket'])  # One-Hot Encoding for bucketized YearMade
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Reduce dataset size (Optional: Use stratified sampling for large datasets)
df_sample = df.sample(frac=0.5, random_state=42)  # Reduce to 50% of data for faster training

# Define features and target
X = df_sample.drop(columns=['SalePrice'])
y = df_sample['SalePrice']

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rf, param_dist, cv=3, n_iter=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Train best model
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())

# Predict and evaluate
preds = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': best_rf.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Save prepared dataset
prepared_data_path = r"C:\Users\eitanb\Documents\DS\ML\ML_project\DATA\result\prepared_dataset.csv"
df_sample.to_csv(prepared_data_path, index=False)

print(f"Prepared dataset saved to {prepared_data_path}")
print(f"Best Parameters: {random_search.best_params_}")
print(f"Cross-Validation RMSE: {cv_rmse}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print("Top 10 Important Features:")
print(feature_importance.head(10))

def RMSLE(y_test, y_pred):
    '''
    RSMLE approximates the percent change
    '''
    return np.sqrt(np.mean((np.log(y_pred) - np.log(y_test))**2))

def RMSE(y_, y_pred_):
    '''
    RSME
    '''
    return ((y_ - y_pred_) ** 2).mean() ** 0.5
# 7. print the RMSE accuracy of the baseline (std dev)
print("RMSE Baseline accuracy:", y_test.std())
print("Train RMSE:", RMSE(y_train, preds))
print("Test RMSE:", RMSE(y_test, preds))
print("Test RMSLE:", RMSLE(y_test, preds))
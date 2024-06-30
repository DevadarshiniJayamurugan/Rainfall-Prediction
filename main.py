import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('C:\\Users\\devad\\Downloads\\Projects Personal\\Rainfall Prediction XGBoost\\rainfall.csv')

# Show initial information
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print(df.columns)

# Strip any leading/trailing spaces from column names
df.rename(str.strip, axis='columns', inplace=True)
print(df.columns)

# Fill missing values with the mean of the column
for col in df.columns:
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col] = df[col].fillna(val)
print(df.isnull().sum().sum())

# Assuming 'sep' represents rainfall (replace 'sep' with the correct column if needed)
rainfall_column = 'sep'

# Plotting the pie chart for the assumed 'rainfall' column
plt.pie(df[rainfall_column].value_counts().values,
        labels = df[rainfall_column].value_counts().index,
        autopct='%1.1f%%')
plt.show()

# Group by the rainfall column and calculate the mean for numeric columns only
print(df.groupby(rainfall_column).mean(numeric_only=True))

# Identify numeric features
features = list(df.select_dtypes(include=np.number).columns)
features.remove(rainfall_column)  # Adjust this based on your dataset
print(features)

# Plot distributions of numeric features
plt.subplots(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.histplot(df[col], kde=True)
plt.tight_layout()
plt.show()

# Plot boxplots of numeric features
plt.subplots(figsize=(15, 8))
for i, col in enumerate(features):
    plt.subplot(3, 4, i + 1)
    sb.boxplot(x=df[col])
plt.tight_layout()
plt.show()

# Encode categorical data if necessary
df.replace({'yes': 1, 'no': 0}, inplace=True)

# Plot the heatmap of correlations
plt.figure(figsize=(10, 10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# Drop features based on high correlation or irrelevance
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

# Define features and target
features = df.drop(['num', rainfall_column], axis=1)  # Adjust 'num' based on your dataset
target = df[rainfall_column]

# Split the dataset
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)

# Balance the dataset
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Train and evaluate models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for model in models:
    model.fit(X, Y)
    print(f'{model} : ')
    train_preds = model.predict_proba(X) 
    print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:, 1]))
    val_preds = model.predict_proba(X_val) 
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:, 1]))
    print()
    metrics.plot_confusion_matrix(model, X_val, Y_val)
    plt.show()

print(metrics.classification_report(Y_val, models[2].predict(X_val)))

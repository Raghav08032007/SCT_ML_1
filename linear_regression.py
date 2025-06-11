import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# === Step 1: Load datasets ===
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# === Step 2: Select features and target ===
features = ['FullBath', 'BedroomAbvGr', 'LotArea']
X = train_df[features].fillna(0)
y = train_df['SalePrice']
test_X = test_df[features].fillna(0)

# === Step 3: Train-test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train Linear Regression model ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Step 5: Evaluate with RMSE ===
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(" Validation RMSE:", round(rmse, 2))

# === Step 6: Correlation Analysis ===
selected_features = features + ['SalePrice']
correlation_df = train_df[selected_features].corr()
print("\n Correlation with SalePrice:\n")
print(correlation_df['SalePrice'].sort_values(ascending=False))

# === Step 7: Heatmap ===
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Bath, Bed, LotArea vs SalePrice")
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

# === Step 8: Scatter Plots ===
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.scatterplot(x='FullBath', y='SalePrice', data=train_df)
plt.title("Bathrooms vs SalePrice")

plt.subplot(1, 3, 2)
sns.scatterplot(x='BedroomAbvGr', y='SalePrice', data=train_df)
plt.title("Bedrooms vs SalePrice")

plt.subplot(1, 3, 3)
sns.scatterplot(x='LotArea', y='SalePrice', data=train_df)
plt.title("Lot Area vs SalePrice")

plt.tight_layout()
plt.savefig("scatter_plots.png")
plt.show()

# === Step 9: Predict test data ===
test_preds = model.predict(test_X)

# === Step 10: Create submission file ===
submission_df = pd.read_csv("sample_submission.csv")
submission_df['SalePrice'] = test_preds
submission_df.to_csv("linear_regression_submission.csv", index=False)
print(" Submission file saved as 'linear_regression_submission.csv'")

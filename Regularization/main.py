import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Read data from CSV file
file = pd.read_csv('communities.data', header=None, sep=',')

# Replace '?' with missing values and drop columns with missing values
file.replace('?', pd.NA, inplace=True)
file.dropna(axis='columns', inplace=True)
file.drop([3], axis='columns', inplace=True)

# Rename columns for better readability
column_names = [f'col_{i}' for i in range(len(file.columns))]
file.columns = column_names

# Split data into training and testing sets
data_train, data_test = train_test_split(file, test_size=0.2)

# Extract target variable (y) and features (X) from training and testing sets
y_train = pd.DataFrame(data_train['col_101'])
x_train = pd.DataFrame(data_train[file.columns[:-1]])
y_expected = pd.DataFrame(data_test['col_101'])
x_test = pd.DataFrame(data_test[file.columns[:-1]])

# Standardize features using StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train a Linear Regression model without regularization
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_predicted = model.predict(x_test_scaled)

# Calculate the Root Mean Squared Error (RMSE) without regularization
rmse = mean_squared_error(y_expected, y_predicted)
print(f"RMSE without regularization: {rmse}")

# Train a Ridge Regression model with regularization (alpha=1.0)
model = Ridge(alpha=1.0)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

# Calculate the Root Mean Squared Error (RMSE) with regularization
rmse = mean_squared_error(y_expected, y_pred)
print(f"RSME with regularization: {rmse}")

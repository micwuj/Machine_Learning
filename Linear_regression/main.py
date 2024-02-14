import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the hypothesis function h(theta, X)
def h(theta, X):
    return theta[0] + theta[1] * X

# Function to normalize data
def normalize_data(data):
    return (data - data.mean()) / data.std()

# Function for linear regression
def linear_regression(theta, X, y, alpha, epsilon):
    m = len(y)
    cost_history = []

    # Gradient Descent iterations
    for _ in range(m):
        error = h(theta, X) - y
        theta[0] -= alpha * (1 / m) * np.sum(error)
        theta[1] -= alpha * (1 / m) * np.sum(error * X)
        cost = (1 / (2 * m)) * np.sum(error**2)
        cost_history.append(cost)

        # Check for convergence
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < epsilon:
            break

    return theta, cost_history

# Read data from file
data = pd.read_csv('fires_thefts.csv', header=None)
fires = data.iloc[:, 0]
thefts = data.iloc[:, 1]
theta = np.zeros(2)

# Normalize the data
fires = normalize_data(fires)
thefts = normalize_data(thefts)

# Set hyperparameters for linear regression
alpha = 0.01
epsilon = 0.0001

theta, cost_history = linear_regression(theta, fires, thefts, alpha, epsilon)

print(f'Theta_0 = {theta[0]}')
print(f'Theta_1 = {theta[1]}\n')

# Plot the cost history over steps
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.show()

# Predict thefts for given number of fires
fires_to_predict = [50, 100, 200]
for fires_count in fires_to_predict:
    predicted_thefts = h(theta, (fires_count - fires.mean()) / fires.std())
    print(f'{fires_count} fires: {predicted_thefts:.2f} thefts')

epsilon = 0.00001
alphas = [0.0001, 0.001, 0.01, 0.1]

# Plot cost history for each alpha value
for x in alphas:
    theta, cost_history = linear_regression(theta, fires, thefts, x, epsilon)
    l = len(cost_history)
    plt.plot(range(l), cost_history, label=f'a = {x}')

plt.xlabel('Steps')
plt.ylabel('Cost')
plt.legend()
plt.show()

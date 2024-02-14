import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def plot(X, Y, degrees):
    """
    Plot polynomial regression curves of different degrees.

    Parameters:
    - X: Independent variable
    - Y: Dependent variable
    - degrees: List of polynomial degrees to visualize
    """

    for degree in degrees:
        # Create a polynomial regression model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
        # Fit the model to the data
        model.fit(X, Y)
        
        # Generate points for plotting the regression curve
        X_plot = np.linspace(min(X), max(X), 500)
        Y_plot = model.predict(X_plot)
        
        # Plot the regression curve for the current degree
        plt.plot(X_plot, Y_plot, label=f'Degree: {degree}')
    
    plt.scatter(X, Y, color='grey')
    
    plt.title(f'Polynomial Regression Degrees: {degrees}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.legend()
    plt.show()

# Read data from a file
alldata = pd.read_csv(
    "data6.tsv",
    header=None,
    sep="\t",
    names=["X", "Y"],
)

# Extract X and Y values from the DataFrame
X = alldata['X'].values.reshape(-1, 1)
Y = alldata['Y'].values

plot(X, Y, [1, 2, 5])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.loss_history = []
        self.steps_history = [] 


    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
            
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept
        
        
        

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)
            

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session
        
        #X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        X_bias=X
        
        
        X_T=X_bias.T
        
        X_T_X = X_T @ X_bias
        
        
        
        if np.linalg.det(X_T_X) != 0:
            X_T_X_inv = np.linalg.inv(X_T_X)
            
        else:  
            X_T_X_inv = np.linalg.pinv(X_T_X)
        
        
        betas = np.dot(np.dot(X_T_X_inv, X_T), y)
        
        self.intercept=betas[0]
        self.coefficients=betas[1:]
        

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
    
        m = len(y)    
        
        self.coefficients = np.random.rand(X.shape[1] - 1) * 0.01 
        
        self.intercept = np.random.rand() * 0.01

        # Implement gradient descent (TODO)
        
        for epoch in range(iterations):
            
            predictions = self.predict(X)
            print(predictions)
            error = predictions - y
            
            
            
            print(error)
            # TODO: Write the gradient values and the updates for the paramenters
            
            gradient_intercept=np.mean(error)
            gradient_coefficients = np.dot(X.T, error) / m
            
            
            self.intercept -= learning_rate * gradient_intercept
            self.coefficients -= learning_rate * gradient_coefficients[1:]

            
            
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)
            self.steps_history.append((self.coefficients.copy(), self.intercept))
            
            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        
        
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        X_bias=None
        
        
        
        # Paste your code from last week
        if  X.ndim == 1 :
            x_reshaped=X.reshape(-1,1)
            
            X_bias = np.insert(x_reshaped, 0, 1, axis=1)    
            predictions = self.intercept + self.coefficients * X  
            
        
        else:
            
            
            
            X_bias = np.insert(X, 0, 1, axis=1)    
            
            params = np.concatenate([np.array([self.intercept]), self.coefficients])
            
            predictions = X_bias @ params

        return predictions



def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """


    
    # R^2 Score
    # TODO
    
    
    suma_residuos = np.sum((y_true - y_pred) ** 2)
    
    
    #suma total de las diferencias al cuadrado entre los valores reales y su media
    
    suma_total = np.sum((y_true - np.mean(y_true)) ** 2)
    
    #calcular el R², restando a 1 la suma de los residuos al cuadrado entre la suma de las diferencias al cuadrado entre los valores reales y su media
    r_squared= 1-(suma_residuos/suma_total)

    # Root Mean Squared Error
    # TODO
    mse=np.mean((y_true-y_pred)**2)
    rmse=np.sqrt(mse)

    # Mean Absolute Error
    # TODO
    mae=np.mean(np.abs(y_true-y_pred))
    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    
    X_transformed = X.copy()
    
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X[:, index]
        
        # TODO: Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = np.array([categorical_column == value for value in unique_values]).T

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed= np.hstack((X_transformed[:, :index], one_hot, X_transformed[:, index:]))
        

    return X_transformed

"""
x = np.array([0, 3, 2, 1, 4, 6, 7, 8, 9, 10])
y = np.array([2, 3, 2, 4, 5, 7, 9, 9, 10, 13])

data = pd.read_csv("data/synthetic_dataset.csv")

# TODO: Obtain inputs and output from data
X1 = data.iloc[:, :-1].values
y1 = data.iloc[:, -1].values



model=LinearRegressor()

model.fit(X1,y1)
y_pred=model.predict(X1)




url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
data = pd.read_csv(url)

# Preprocess the data
# TODO: One-hot encode categorical variables. Use pd.get_dummies()
data_encoded = pd.get_dummies(data, drop_first=True) 

# Split the data into features (X) and target (y)
X = data_encoded.drop('charges', axis=1)
y = data_encoded['charges']

print(X.shape)

# Instantiate the LinearRegression model
model=LinearRegressor()

# Fit the model on the training data
model.fit_gradient_descent(X, y)

# Make predictions on the test data
y_pred = model.predict(X)

# Evaluate the model
evaluation_metrics = evaluate_regression(y, y_pred)
print(evaluation_metrics)
"""
# MiniBatchGradientDescentRegression
# Mini-Batch Gradient Descent Regression
### The MiniBatchGDRegression class is an implementation of linear regression using the mini-batch gradient descent optimization algorithm. Linear regression is a fundamental supervised learning technique used to model the relationship between a dependent variable and one or more independent variables. It aims to find the best-fitting linear equation through the data points to make predictions.
## Mathematical Formulation:
### In linear regression, the relationship between the dependent variable y and the independent variables (features) X is represented as:

$\huge{\mathbf{y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n}}$

## where:
### • y is the target (dependent) variable.
### • $\mathbf{x}$ is the vector of input (independent) variables.
### • $\mathbf{\beta}$ is the vector of coefficients, representing the weights of the model for each feature.
### • $\beta_0$ is the intercept, the value of y when all features are 0.
### The goal of linear regression is to determine the optimal values of $\mathbf{\beta}$ and $\beta_0$ that minimize the difference between the predicted values and the actual target values.
## Mini-Batch Gradient Descent and Vectorized Operations:
### The provided implementation employs mini-batch gradient descent for efficient optimization. Mini-batch gradient descent divides the dataset into smaller batches and updates the model's parameters iteratively using these batches. This approach strikes a balance between the computational efficiency of stochastic gradient descent and the stability of batch gradient descent.
### In the context of mini-batch gradient descent, vectorized operations are employed to improve computational efficiency. NumPy, a popular numerical library in Python, facilitates these operations, enabling simultaneous mathematical computations on entire arrays.
## Update Equations (Vectorized):
### 1) Update Intercept $\beta_0$:
$\huge{\beta_0 \leftarrow \beta_0 - \alpha \frac{1}{m} \sum (y - \hat{y})}$

## Vectorized form:
# $\beta_0 \leftarrow \beta_0 - \alpha \frac{1}{m} (\mathbf{y} - \mathbf{X} \mathbf{\beta})$

### 2) Update Coefficients $\mathbf{\beta}$:
# $\mathbf{\beta} \leftarrow \mathbf{\beta} - \alpha \frac{1}{m} \sum ((y - \hat{y}) \cdot \mathbf{x})$

## Vectorized form:
$\Huge{\mathbf{\beta}} \leftarrow \Huge{\mathbf{\beta}} - \Huge{\alpha} \frac{\Huge{1}}{\Huge{m}} (\Huge{\mathbf{X}}^T (\Huge{\mathbf{y}} - \Huge{\mathbf{X}} \Huge{\mathbf{\beta}}))$


## where:
### • $\mathbf{X}$ is the matrix of input features with shape (n_samples, n_features).

### • $\mathbf{y}$ is the vector of actual target values.
### • $\alpha$ (learning rate) is a hyperparameter controlling the step size in each iteration.
### • $m$ is the number of samples in the training data.
### • $\mathbf{\beta}$ is the vector of model parameters, including the intercept and coefficients.
## How to Use:
### To utilize this implementation, proceed with the following steps:
## 1) Instantiate the Model:
### Create an instance of the MiniBatchGDRegression class.
## 2) Fit the Model:
### Train the model on your training data using the fit method. Provide the training features and target values along with optional parameters such as batch size (batch_size), learning rate (lr), and maximum iterations (max_iteration).
## 3) Make Predictions:
### Employ the predict method to make predictions on new data. Provide the input data with shape (n_samples, n_features), and the method will yield the predicted target values.
## 4) Evaluate the Model:
### Assess the model's performance using the score method, which calculates the coefficient of determination ($R^2$ Score). The $R^2$ Score gauges the goodness of fit by measuring the agreement between predicted and actual target values.

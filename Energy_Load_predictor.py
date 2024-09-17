# Manav Sharma
# R00183839
# Note: I have provided a detailed explanation of the following code in the "Final Assignment Explanation.Docx"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error


# Load the data from the CSV file
data = pd.read_csv('energy_performance.csv')

# Extract features and targets
features = data[['Relative compactness', 'Surface area', 'Wall area', 'Roof area', 'Overall height', 'Orientation', 'Glazing area', 'Glazing area distribution']]
targets = data[['Heating load', 'Cooling load']]


def task1():

    # Determine and output min/max heating and cooling loads
    min_heating_load = targets['Heating load'].min()
    max_heating_load = targets['Heating load'].max()
    min_cooling_load = targets['Cooling load'].min()
    max_cooling_load = targets['Cooling load'].max()

    print(f"Min Heating Load: {min_heating_load}")
    print(f"Max Heating Load: {max_heating_load}")
    print(f"Min Cooling Load: {min_cooling_load}")
    print(f"Max Cooling Load: {max_cooling_load}")

# task1()

# ---------TASK 2-----------

def calculate_polynomial_model(degree, features, coefficients):
    """
    Parameters:
    - degree: int, the degree of the polynomial
    - features: array-like, list of feature vectors
    - coefficients: array-like, parameter vector of coefficients
    """

    # Create a feature matrix with polynomial terms
    feature_matrix = np.column_stack([np.power(features, d) for d in range(degree + 1)])

    # Calculate the dot product of the feature matrix and coefficients
    return np.dot(feature_matrix, coefficients)


def determine_parameter_vector_size(degree, num_features):
    """
    Parameters:
    - degree: int, the degree of the polynomial
    - num_features: int, the number of features
    """
    # The size is determined by (degree + 1) times the number of features
    return (degree + 1) * num_features


def task2(degree):
    """
    Parameters:
    - degree: int, the degree of the polynomial
    """

    # Determine the number of features, excluding the intercept term
    num_features = features.shape[1]

    # Extract feature vectors from the DataFrame
    features_array = features.values

    # Initialize coefficients with random values
    coefficients = np.random.rand(determine_parameter_vector_size(degree, num_features))

    # Calculate the estimated target vector
    estimated_target = calculate_polynomial_model(degree, features_array, coefficients)

    # Print the estimated target vector
    print("Estimated Target:", estimated_target)


# Example: Assuming a polynomial of degree 2
# task2(degree=2)


# -----------TASK 3-----------------

def calculate_model_function(features, coefficients):
    degree = len(coefficients) - 1
    result = np.zeros(features.shape[0])
    for i in range(degree + 1):
        result += coefficients[i] * np.prod(features**i, axis=1)
    return result


def calculate_jacobian(degree, feature_vectors, coefficients):
    num_features = feature_vectors.shape[1]
    num_samples = feature_vectors.shape[0]
    jacobian = np.zeros((num_samples, num_features * (degree + 1)))

    for i in range(degree + 1):
        jacobian[:, i * num_features: (i + 1) * num_features] = feature_vectors**i

    estimated_target = calculate_model_function(feature_vectors, coefficients)

    return estimated_target, jacobian


def linearize(feature_vectors, coefficients, degree):
    estimated_target, jacobian = calculate_jacobian(degree, feature_vectors, coefficients)
    return estimated_target, jacobian


def task3():
    degree = 3
    feature_vectors = np.array([[146253.04307569, 146257.36740222, 146263.38793635], [174821.40154461, 174825.72587115, 174831.74640527]])
    coefficients = np.array([1.0, 2.0, 3.0, 4.0])  # These coefficients are taken as examples

    estimated_target, jacobian = linearize(feature_vectors, coefficients, degree)

    print("Estimated Target:")
    print(estimated_target)
    print("\nJacobian:")
    print(jacobian)


# task3()


# -----------TASK 4-----------------

def calculate_optimal_parameter_update(training_targets, estimated_targets, jacobian):
    # Calculate the normal equation matrix
    normal_eq_matrix = np.dot(jacobian.T, jacobian)

    # Add regularization term to prevent singularity
    reg_lambda = 0.01  # we can adjust the regularization parameter
    reg_matrix = reg_lambda * np.identity(normal_eq_matrix.shape[0])
    normal_eq_matrix += reg_matrix

    # Calculate residuals
    residuals = training_targets - estimated_targets

    # Build the normal equation system
    right_hand_side = np.dot(jacobian.T, residuals)

    # Solve the normal equation system to obtain the optimal parameter update
    optimal_parameter_update = np.linalg.solve(normal_eq_matrix, right_hand_side)

    return optimal_parameter_update


# ----------TASK 5 ---------------------


def regression(degree, features, training_targets, num_iterations=100, tol=1e-5):
    # Initialize the parameter vector of coefficients with zeros
    num_features = features.shape[1]
    coefficients = np.zeros(determine_parameter_vector_size(degree, num_features))

    for iteration in range(num_iterations):
        # Linearize at the current coefficients
        estimated_targets, jacobian = linearize(features, coefficients, degree)

        # Calculate the optimal parameter update
        optimal_parameter_update = calculate_optimal_parameter_update(training_targets, estimated_targets, jacobian)

        # Update the coefficients
        coefficients += optimal_parameter_update

        # Check for convergence
        if np.linalg.norm(optimal_parameter_update) < tol:
            break

    return coefficients


# --------------TASK 6 ----------------


def model_selection(degrees, features, targets, load_type):
    """
    Parameters:
    - degrees: list, polynomial degrees to be evaluated
    - features: array-like, input features
    - targets: array-like, target values
    - load_type: str, either 'Heating' or 'Cooling' to specify the load type
    """
    optimal_degree = None
    min_mean_absolute_diff = float('inf')

    for degree in degrees:
        # Use cross_val_predict to get predicted values for each fold
        predicted_values = cross_val_predict(degree, features, targets[load_type + ' load'], cv=5)

        # Calculate mean absolute differences for each fold
        mean_absolute_diff = np.mean(np.abs(targets[load_type + ' load'] - predicted_values))

        # Update optimal degree if the current degree has a lower mean absolute difference
        if mean_absolute_diff < min_mean_absolute_diff:
            min_mean_absolute_diff = mean_absolute_diff
            optimal_degree = degree

    return optimal_degree, min_mean_absolute_diff


# Example usage
degrees_to_evaluate = [0, 1, 2]
optimal_degree_heating, mean_absolute_diff_heating = model_selection(degrees_to_evaluate, features, targets, 'Heating')
optimal_degree_cooling, mean_absolute_diff_cooling = model_selection(degrees_to_evaluate, features, targets, 'Cooling')

print(f"Optimal Degree for Heating: {optimal_degree_heating}")
print(f"Mean Absolute Difference for Heating: {mean_absolute_diff_heating}")

print(f"Optimal Degree for Cooling: {optimal_degree_cooling}")
print(f"Mean Absolute Difference for Cooling: {mean_absolute_diff_cooling}")


# --------------TASK 7 ----------------


def evaluate_and_visualize(degrees, features, targets):

    optimal_degree_heating, mean_absolute_diff_heating = model_selection(degrees, features, targets[['Heating load']])
    optimal_degree_cooling, mean_absolute_diff_cooling = model_selection(degrees, features, targets[['Cooling load']])

    # Estimate model parameters for both heating and cooling loads using the selected optimal model function
    coefficients_heating = regression(optimal_degree_heating, features, targets['Heating load'])
    coefficients_cooling = regression(optimal_degree_cooling, features, targets['Cooling load'])

    # Calculate predicted heating and cooling loads
    predicted_heating = calculate_model_function(features.values, coefficients_heating)
    predicted_cooling = calculate_model_function(features.values, coefficients_cooling)

    # Plot estimated loads against true loads
    plt.scatter(targets['Heating load'], predicted_heating, label='Heating Load', alpha=0.5)
    plt.scatter(targets['Cooling load'], predicted_cooling, label='Cooling Load', alpha=0.5)
    plt.xlabel('True Load')
    plt.ylabel('Estimated Load')
    plt.legend()
    plt.show()

    # Output mean absolute difference

    print(f"Mean Absolute Difference (Heating): {mean_absolute_diff_heating}")
    print(f"Mean Absolute Difference (Cooling): {mean_absolute_diff_cooling}")


# Example usage
degrees_to_evaluate = [0, 1, 2]
evaluate_and_visualize(degrees_to_evaluate, features, targets)


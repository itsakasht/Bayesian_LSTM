import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
def objective_function(x):
    return (x - 3)*(x-6)  # True function we aim to optimize

# Define the noisy simulation of the objective function
def simulate_function(x):
    noise = np.random.normal(0, 1)  # Add Gaussian noise
    return objective_function(x) + noise

# Define the distance function for ABC
def distance(simulated, observed):
    return abs(simulated - observed)

# ABC Rejection Sampling
def abc_optimization(observed_value, prior_range, num_samples, epsilon):
    """
    Parameters:
    - observed_value: The target value of the function (simulated minimum).
    - prior_range: Range for sampling parameters (uniform prior).
    - num_samples: Number of samples to generate.
    - epsilon: Tolerance for acceptance.
    """
    accepted_params = []
    
    for _ in range(num_samples):
        # Sample from the prior (uniform distribution in this example)
        candidate_param = np.random.uniform(prior_range[0], prior_range[1])
        
        # Simulate the function with the candidate parameter
        simulated_value = simulate_function(candidate_param)
        
        # Calculate the distance
        if distance(simulated_value, observed_value) <= epsilon:
            accepted_params.append(candidate_param)
    
    return np.array(accepted_params)

# Parameters for ABC
observed_value = 0  # True minimum value of the function is f(3) = 0
prior_range = (0, 10)  # Search for x in the range [0, 6]
num_samples = 10000  # Number of samples to generate
epsilon = 1  # Tolerance for acceptance

# Perform ABC optimization
accepted_params = abc_optimization(observed_value, prior_range, num_samples, epsilon)

# Results
print(f"Number of accepted parameters: {len(accepted_params)}")
print(f"Mean of accepted parameters: {np.mean(accepted_params):.3f}")
print(f"Standard deviation of accepted parameters: {np.std(accepted_params):.3f}")

# Visualization
x_values = np.linspace(prior_range[0], prior_range[1], 500)
y_values = objective_function(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label="True Function: ", color="blue")
plt.hist(accepted_params, bins=30, density=True, alpha=0.5, label="Accepted Parameters")
plt.axvline(x=np.mean(accepted_params), color='red', linestyle='--', label="Mean of Accepted Params")
plt.title("ABC Optimization - Parameter Distribution")
plt.xlabel("Parameter x")
plt.ylabel("Frequency / Function Value")
plt.legend()
plt.show()
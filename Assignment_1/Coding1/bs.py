import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
t = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X]  # Shape (100,2)

# Hyperparameters
alpha = 0.1  # Learning rate
iterations = 1000  # Number of iterations
m = len(t)  # Number of training examples

# Initialize w randomly
w = np.random.randn(2, 1)

# Perform gradient descent
for i in range(iterations):
    gradients = (2/m) * X_b.T.dot(X_b.dot(w) - t)  # Compute gradient
    w -= alpha * gradients  # Update weights

# Print final weights
print("Final weights:", w.ravel())

# Compare with normal equation
w_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(t)
print("Normal equation solution:", w_normal.ravel())

# Plot the results
plt.scatter(X, t, color="blue", label="Data")
plt.plot(X, X_b.dot(w), color="red", label="Gradient Descent")
plt.plot(X, X_b.dot(w_normal), color="green", linestyle="dashed", label="Normal Equation")
plt.xlabel("X")
plt.ylabel("t")
plt.legend()
plt.show()

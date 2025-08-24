import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate white noise (stationary)
n_points = 1000
white_noise = np.random.normal(loc=0, scale=1, size=n_points)

# Moving Average (MA) process of order 3: y_t = x_t + 0.5*x_{t-1} + 0.25*x_{t-2}
ma_order = 2
ma_signal = np.zeros(n_points)
for t in range(ma_order, n_points):
    ma_signal[t] = white_noise[t] + 0.5 * white_noise[t-1]

# Plot the MA signal
plt.figure(figsize=(12, 6))
plt.plot(ma_signal, label="Stationary MA Process Signal", color="blue")
plt.title("Stationary Moving Average (MA) Process Signal")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

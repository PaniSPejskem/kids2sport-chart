import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize

# Input data
percentiles = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
values = np.array([0.6, 0.8, 0.9, 1.2, 1.4, 1.7, 2.0, 2.4, 2.9, 3.4, 4.2, 5.0, 6.4])
cum_probs = percentiles / 100.0

# --- Quantile-based log-normal fitting ---

# Objective function: minimize squared error between model percentiles and actual data
def quantile_fit_loss(params):
    shape, scale = params
    predicted = lognorm.ppf(cum_probs, shape, loc=0, scale=scale)
    return np.sum((predicted - values) ** 2)

# Initial guess (based on log-space std & mean)
init_shape, _, init_scale = lognorm.fit(values, floc=0)

# Optimize
res = minimize(quantile_fit_loss, x0=[init_shape, init_scale], bounds=[(0.01, 5), (0.1, 10)])
shape_opt, scale_opt = res.x

# Generate model curve
percentiles_smooth = np.linspace(1, 99, 500)
cum_probs_smooth = percentiles_smooth / 100
fitted_values = lognorm.ppf(cum_probs_smooth, shape_opt, loc=0, scale=scale_opt)

# Create subplots
fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(21, 6))

# Plot 1: Percentile vs Shuttle Run Value
ax1.plot(percentiles_smooth, fitted_values, label='Fitted Log-Normal (Quantile Fit)', color='blue')
ax1.scatter(percentiles, values, color='red', label='Empirical Data', zorder=5)
ax1.set_title('Shuttle Run Value vs Percentile (Log-Normal Quantile Fit)')
ax1.set_xlabel('Percentile')
ax1.set_ylabel('Shuttle Run Value')
ax1.grid(True)
ax1.legend()

# Plot 2: Probability Density Function (PDF) of the fitted log-normal
shuttle_values_smooth = np.linspace(0.5, 7.0, 500)
pdf_values = lognorm.pdf(shuttle_values_smooth, shape_opt, loc=0, scale=scale_opt)
ax2.plot(shuttle_values_smooth, pdf_values, label='Fitted Log-Normal PDF', color='green', linewidth=2)
ax2.set_title('Probability Density of Shuttle Run Values')
ax2.set_xlabel('Shuttle Run Value')
ax2.set_ylabel('Probability Density')
ax2.grid(True)
ax2.legend()

# Plot 3: Empirical CDF vs Model CDF
ax4.step(values, cum_probs, where='post', color='red', label='Empirical CDF', linewidth=2)
model_cdf = lognorm.cdf(shuttle_values_smooth, shape_opt, loc=0, scale=scale_opt)
ax4.plot(shuttle_values_smooth, model_cdf, color='green', label='Fitted Log-Normal CDF', linewidth=2)
ax4.set_title('Empirical CDF vs Fitted CDF')
ax4.set_xlabel('Shuttle Run Value')
ax4.set_ylabel('Cumulative Probability')
ax4.grid(True)
ax4.legend()

# Adjust layout for three plots
fig.set_size_inches(21, 6)
plt.tight_layout()
plt.show()

# Print the fitted log-normal CDF function and parameters
print("\nFitted Log-Normal CDF:")
print("CDF(x) = lognorm.cdf(x, shape, loc=0, scale=scale)")
print(f"shape = {shape_opt:.6f}")
print(f"scale = {scale_opt:.6f}")
print("loc = 0 (fixed)")

# Usage: user input value, output percentile
user_value = float(input("\nEnter a shuttle run value to get the percentile: "))
percentile = lognorm.cdf(user_value, shape_opt, loc=0, scale=scale_opt) * 100
print(f"Percentile for value {user_value}: {percentile:.2f} %")

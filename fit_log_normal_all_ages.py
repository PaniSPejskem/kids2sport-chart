import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize
from ipywidgets import interact, Dropdown
import ipywidgets as widgets
from matplotlib.widgets import Button
from scipy.interpolate import interp1d

# --- CONFIG ---
input_csv = 'data/males_sit_and_reach.csv'  # Change as needed
output_dir = 'data/output'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, os.path.basename(input_csv))

# --- PARAMETERS ---
ERROR_THRESHOLD = 0.1  # Adjust as needed (relative squared error)

# --- LOAD DATA ---
def load_percentile_data(csv_path):
    df = pd.read_csv(csv_path, sep='\t', decimal=',')
    percentiles = [int(x) for x in df.columns[1:]]
    ages = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1:].values.astype(float)
    return percentiles, ages, values

percentiles, ages, values_matrix = load_percentile_data(input_csv)
cum_probs = np.array(percentiles) / 100.0

# --- DETECT AND HANDLE INVERTED DATA ---
def detect_and_fix_inverted_data(values_matrix, cum_probs):
    """
    Detect if data is inverted (higher percentiles have lower values).
    If inverted, flip the data and return True, otherwise return False.
    """
    # Check if the trend is inverted by comparing first and last percentiles
    # across all age groups
    inverted_count = 0
    total_checks = 0
    
    for row in values_matrix:
        if len(row) >= 2:  # Need at least 2 points to check trend
            first_val = row[0]  # 1st percentile value
            last_val = row[-1]  # 99th percentile value
            if first_val > last_val:  # Higher percentile has lower value = inverted
                inverted_count += 1
            total_checks += 1
    
    # If majority of rows show inverted trend, consider data inverted
    is_inverted = inverted_count > total_checks / 2
    
    if is_inverted:
        print("WARNING: Data appears to be inverted (higher percentiles have lower values).")
        print("This is unusual for performance metrics. Inverting data to correct format.")
        print("If this is incorrect, please check your data source.")
        
        # Invert the data by flipping the values array
        values_matrix_inverted = np.flip(values_matrix, axis=1)
        return values_matrix_inverted, True
    else:
        return values_matrix, False

# Apply inversion detection and correction
values_matrix, data_was_inverted = detect_and_fix_inverted_data(values_matrix, cum_probs)

# --- FITTING ---
def fit_lognorm_percentiles(values, cum_probs):
    def quantile_fit_loss(params):
        shape, scale = params
        predicted = lognorm.ppf(cum_probs, shape, loc=0, scale=scale)
        return np.sum((predicted - values) ** 2)
    init_shape, _, init_scale = lognorm.fit(values, floc=0)
    res = minimize(quantile_fit_loss, x0=[init_shape, init_scale], bounds=[(0.01, 5), (0.1, 20)])
    shape_opt, scale_opt = res.x
    return shape_opt, scale_opt

fit_results = []
fit_methods = []
fit_errors = []
for i, row in enumerate(values_matrix):
    # Fallback to linear interpolation if any value is <= 0
    if np.any(row <= 0):
        print(f"WARNING: Age {ages[i]} contains zero or negative values. Using linear interpolation instead of log-normal fit.")
        shape, scale = np.nan, np.nan
        fit_method = 'linear_interp'
        error = np.nan
    else:
        shape, scale = fit_lognorm_percentiles(row, cum_probs)
        # Calculate fit error (relative squared error)
        predicted = lognorm.ppf(cum_probs, shape, loc=0, scale=scale)
        error = np.sum((predicted - row) ** 2) / np.sum(row ** 2)
        if error > ERROR_THRESHOLD:
            fit_method = 'linear_interp'
        else:
            fit_method = 'lognorm'
    fit_results.append((ages[i], shape, scale, 0, data_was_inverted, fit_method))
    fit_methods.append(fit_method)
    fit_errors.append(error)

# --- SAVE OUTPUT CSV ---
output_df = pd.DataFrame(fit_results, columns=['age', 'shape', 'scale', 'loc', 'invert_data', 'fit_method'])
output_df.to_csv(output_csv, index=False)

# --- PRINT TO CONSOLE ---
print('age,shape,scale,loc,invert_data,fit_method,fit_error')
for (age, shape, scale, loc, invert_flag, fit_method), error in zip(fit_results, fit_errors):
    print(f'{age},{shape:.6f},{scale:.6f},{loc},{invert_flag},{fit_method},{error:.6f}')

# --- EXPORT IMAGES FOR ALL AGES ---
export_dir = 'data/output/images'
os.makedirs(export_dir, exist_ok=True)

def save_age_plot(age_idx, save_path):
    vals = values_matrix[age_idx]
    shape, scale, loc, invert_flag, fit_method = fit_results[age_idx][1:]
    percentiles_smooth = np.linspace(1, 99, 500)
    cum_probs_smooth = percentiles_smooth / 100
    fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(21, 6))
    # Q-Q plot
    if fit_method == 'lognorm':
        fitted_values = lognorm.ppf(cum_probs_smooth, shape, loc=0, scale=scale)
        ax1.plot(percentiles_smooth, fitted_values, label='Fitted Log-Normal (Quantile Fit)', color='blue')
    else:
        interp_func = interp1d(percentiles, vals, kind='linear', fill_value='extrapolate')
        fitted_values = interp_func(percentiles_smooth)
        ax1.plot(percentiles_smooth, fitted_values, label='Linear Interpolation', color='orange')
    ax1.scatter(percentiles, vals, color='red', label='Empirical Data', zorder=5)
    ax1.set_title(f'Shuttle Run Value vs Percentile (Age {ages[age_idx]})')
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Shuttle Run Value')
    ax1.grid(True)
    ax1.legend()
    # PDF
    min_val, max_val = np.min(vals)*0.8, np.max(vals)*1.2
    shuttle_values_smooth = np.linspace(min_val, max_val, 500)
    if fit_method == 'lognorm':
        pdf_values = lognorm.pdf(shuttle_values_smooth, shape, loc=0, scale=scale)
        ax2.plot(shuttle_values_smooth, pdf_values, label='Fitted Log-Normal PDF', color='green', linewidth=2)
    else:
        ax2.text(0.5, 0.5, 'No PDF for Linear Interp', ha='center', va='center', fontsize=14, color='orange', transform=ax2.transAxes)
    ax2.set_title('Probability Density of Shuttle Run Values')
    ax2.set_xlabel('Shuttle Run Value')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True)
    ax2.legend()
    # CDF
    if fit_method == 'lognorm':
        model_cdf = lognorm.cdf(shuttle_values_smooth, shape, loc=0, scale=scale)
        ax4.plot(shuttle_values_smooth, model_cdf, color='green', label='Fitted Log-Normal CDF', linewidth=2)
    else:
        interp_cdf = interp1d(vals, cum_probs, kind='linear', fill_value=(cum_probs[0], cum_probs[-1]), bounds_error=False)
        cdf_y = interp_cdf(shuttle_values_smooth)
        ax4.plot(shuttle_values_smooth, cdf_y, color='orange', label='Linear Interpolation', linewidth=2)
    ax4.step(vals, cum_probs, where='post', color='red', label='Empirical CDF', linewidth=2)
    ax4.set_title('Empirical CDF vs Fitted CDF')
    ax4.set_xlabel('Shuttle Run Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True)
    ax4.legend()
    # Add note about data inversion or fit method
    if invert_flag:
        fit_note = f'(Data was inverted, {fit_method})'
    else:
        fit_note = f'({fit_method})'
    fig.suptitle(f'Log-Normal/Linear Fit for Age {ages[age_idx]} {fit_note}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Save images for all ages
print(f"\nExporting images to {export_dir}...")
for i, age in enumerate(ages):
    filename = f"age_{age}_fit.png"
    filepath = os.path.join(export_dir, filename)
    save_age_plot(i, filepath)
    print(f"Saved: {filename}")

print(f"All {len(ages)} images exported successfully!")

# --- VISUALIZATION WITH MATPLOTLIB BUTTONS ---
current_age_idx = [0]  # Use a mutable object to allow modification in callbacks

fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(21, 6))
plt.subplots_adjust(bottom=0.2)  # Make space for buttons

# Add buttons
axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
bprev = Button(axprev, 'Previous')
bnext = Button(axnext, 'Next')

# Plotting function
plot_handles = {'fig': fig, 'ax1': ax1, 'ax2': ax2, 'ax4': ax4}
def update_plot(age_idx):
    vals = values_matrix[age_idx]
    shape, scale, loc, invert_flag, fit_method = fit_results[age_idx][1:]
    percentiles_smooth = np.linspace(1, 99, 500)
    cum_probs_smooth = percentiles_smooth / 100
    ax1.clear(); ax2.clear(); ax4.clear()
    # Q-Q plot
    if fit_method == 'lognorm':
        fitted_values = lognorm.ppf(cum_probs_smooth, shape, loc=0, scale=scale)
        ax1.plot(percentiles_smooth, fitted_values, label='Fitted Log-Normal (Quantile Fit)', color='blue')
    else:
        interp_func = interp1d(percentiles, vals, kind='linear', fill_value='extrapolate')
        fitted_values = interp_func(percentiles_smooth)
        ax1.plot(percentiles_smooth, fitted_values, label='Linear Interpolation', color='orange')
    ax1.scatter(percentiles, vals, color='red', label='Empirical Data', zorder=5)
    ax1.set_title(f'Shuttle Run Value vs Percentile (Age {ages[age_idx]})')
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Shuttle Run Value')
    ax1.grid(True)
    ax1.legend()
    # PDF
    min_val, max_val = np.min(vals)*0.8, np.max(vals)*1.2
    shuttle_values_smooth = np.linspace(min_val, max_val, 500)
    if fit_method == 'lognorm':
        pdf_values = lognorm.pdf(shuttle_values_smooth, shape, loc=0, scale=scale)
        ax2.plot(shuttle_values_smooth, pdf_values, label='Fitted Log-Normal PDF', color='green', linewidth=2)
    else:
        ax2.text(0.5, 0.5, 'No PDF for Linear Interp', ha='center', va='center', fontsize=14, color='orange', transform=ax2.transAxes)
    ax2.set_title('Probability Density of Shuttle Run Values')
    ax2.set_xlabel('Shuttle Run Value')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True)
    ax2.legend()
    # CDF
    if fit_method == 'lognorm':
        model_cdf = lognorm.cdf(shuttle_values_smooth, shape, loc=0, scale=scale)
        ax4.plot(shuttle_values_smooth, model_cdf, color='green', label='Fitted Log-Normal CDF', linewidth=2)
    else:
        interp_cdf = interp1d(vals, cum_probs, kind='linear', fill_value=(cum_probs[0], cum_probs[-1]), bounds_error=False)
        cdf_y = interp_cdf(shuttle_values_smooth)
        ax4.plot(shuttle_values_smooth, cdf_y, color='orange', label='Linear Interpolation', linewidth=2)
    ax4.step(vals, cum_probs, where='post', color='red', label='Empirical CDF', linewidth=2)
    ax4.set_title('Empirical CDF vs Fitted CDF')
    ax4.set_xlabel('Shuttle Run Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True)
    ax4.legend()
    # Add note about data inversion or fit method
    if invert_flag:
        fit_note = f'(Data was inverted, {fit_method})'
    else:
        fit_note = f'({fit_method})'
    fig.suptitle(f'Age: {ages[age_idx]} {fit_note}', fontsize=16)
    fig.canvas.draw_idle()

# Button callbacks
def next_age(event):
    if current_age_idx[0] < len(ages) - 1:
        current_age_idx[0] += 1
        update_plot(current_age_idx[0])
def prev_age(event):
    if current_age_idx[0] > 0:
        current_age_idx[0] -= 1
        update_plot(current_age_idx[0])
bnext.on_clicked(next_age)
bprev.on_clicked(prev_age)

# Initial plot
update_plot(current_age_idx[0])
plt.show() 
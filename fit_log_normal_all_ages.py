import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize
from ipywidgets import interact, Dropdown
import ipywidgets as widgets
from matplotlib.widgets import Button

# --- CONFIG ---
input_csv = 'data/males_20_shuttle_run.csv'  # Change as needed
output_dir = 'data/output'
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, os.path.basename(input_csv))

# --- LOAD DATA ---
def load_percentile_data(csv_path):
    df = pd.read_csv(csv_path, sep='\t', decimal=',')
    percentiles = [int(x) for x in df.columns[1:]]
    ages = df.iloc[:, 0].astype(str).tolist()
    values = df.iloc[:, 1:].values.astype(float)
    return percentiles, ages, values

percentiles, ages, values_matrix = load_percentile_data(input_csv)
cum_probs = np.array(percentiles) / 100.0

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
for i, row in enumerate(values_matrix):
    shape, scale = fit_lognorm_percentiles(row, cum_probs)
    fit_results.append((ages[i], shape, scale, 0))  # loc is always 0

# --- SAVE OUTPUT CSV ---
output_df = pd.DataFrame(fit_results, columns=['age', 'shape', 'scale', 'loc'])
output_df.to_csv(output_csv, index=False)

# --- PRINT TO CONSOLE ---
print('age,shape,scale,loc')
for age, shape, scale, loc in fit_results:
    print(f'{age},{shape:.6f},{scale:.6f},{loc}')

# --- EXPORT IMAGES FOR ALL AGES ---
export_dir = 'data/output/images'
os.makedirs(export_dir, exist_ok=True)

def save_age_plot(age_idx, save_path):
    vals = values_matrix[age_idx]
    shape, scale, loc = fit_results[age_idx][1:]
    percentiles_smooth = np.linspace(1, 99, 500)
    cum_probs_smooth = percentiles_smooth / 100
    fitted_values = lognorm.ppf(cum_probs_smooth, shape, loc=0, scale=scale)
    
    fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(21, 6))
    
    # Q-Q plot
    ax1.plot(percentiles_smooth, fitted_values, label='Fitted Log-Normal (Quantile Fit)', color='blue')
    ax1.scatter(percentiles, vals, color='red', label='Empirical Data', zorder=5)
    ax1.set_title(f'Shuttle Run Value vs Percentile (Age {ages[age_idx]})')
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Shuttle Run Value')
    ax1.grid(True)
    ax1.legend()
    
    # PDF
    min_val, max_val = np.min(vals)*0.8, np.max(vals)*1.2
    shuttle_values_smooth = np.linspace(min_val, max_val, 500)
    pdf_values = lognorm.pdf(shuttle_values_smooth, shape, loc=0, scale=scale)
    ax2.plot(shuttle_values_smooth, pdf_values, label='Fitted Log-Normal PDF', color='green', linewidth=2)
    ax2.set_title('Probability Density of Shuttle Run Values')
    ax2.set_xlabel('Shuttle Run Value')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True)
    ax2.legend()
    
    # CDF
    ax4.step(vals, cum_probs, where='post', color='red', label='Empirical CDF', linewidth=2)
    model_cdf = lognorm.cdf(shuttle_values_smooth, shape, loc=0, scale=scale)
    ax4.plot(shuttle_values_smooth, model_cdf, color='green', label='Fitted Log-Normal CDF', linewidth=2)
    ax4.set_title('Empirical CDF vs Fitted CDF')
    ax4.set_xlabel('Shuttle Run Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True)
    ax4.legend()
    
    fig.suptitle(f'Log-Normal Fit for Age {ages[age_idx]}', fontsize=16)
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
    shape, scale, loc = fit_results[age_idx][1:]
    percentiles_smooth = np.linspace(1, 99, 500)
    cum_probs_smooth = percentiles_smooth / 100
    fitted_values = lognorm.ppf(cum_probs_smooth, shape, loc=0, scale=scale)
    ax1.clear(); ax2.clear(); ax4.clear()
    # Q-Q plot
    ax1.plot(percentiles_smooth, fitted_values, label='Fitted Log-Normal (Quantile Fit)', color='blue')
    ax1.scatter(percentiles, vals, color='red', label='Empirical Data', zorder=5)
    ax1.set_title(f'Shuttle Run Value vs Percentile (Age {ages[age_idx]})')
    ax1.set_xlabel('Percentile')
    ax1.set_ylabel('Shuttle Run Value')
    ax1.grid(True)
    ax1.legend()
    # PDF
    min_val, max_val = np.min(vals)*0.8, np.max(vals)*1.2
    shuttle_values_smooth = np.linspace(min_val, max_val, 500)
    pdf_values = lognorm.pdf(shuttle_values_smooth, shape, loc=0, scale=scale)
    ax2.plot(shuttle_values_smooth, pdf_values, label='Fitted Log-Normal PDF', color='green', linewidth=2)
    ax2.set_title('Probability Density of Shuttle Run Values')
    ax2.set_xlabel('Shuttle Run Value')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True)
    ax2.legend()
    # CDF
    ax4.step(vals, cum_probs, where='post', color='red', label='Empirical CDF', linewidth=2)
    model_cdf = lognorm.cdf(shuttle_values_smooth, shape, loc=0, scale=scale)
    ax4.plot(shuttle_values_smooth, model_cdf, color='green', label='Fitted Log-Normal CDF', linewidth=2)
    ax4.set_title('Empirical CDF vs Fitted CDF')
    ax4.set_xlabel('Shuttle Run Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True)
    ax4.legend()
    fig.suptitle(f'Age: {ages[age_idx]}', fontsize=16)
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
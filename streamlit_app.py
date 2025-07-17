import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Log-Normal Distribution Fitter",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Log-Normal Distribution Fitter")
st.markdown("""
Upload a CSV file with percentile data to fit log-normal distributions for each age group.
The app will automatically detect data issues and use appropriate fitting methods.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
ERROR_THRESHOLD = st.sidebar.slider(
    "Error Threshold for Log-Normal Fit", 
    min_value=0.001, 
    max_value=0.5, 
    value=0.1, 
    step=0.001,
    help="If fit error exceeds this threshold, linear interpolation will be used instead"
)

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV file", 
    type=['csv'],
    help="Upload a CSV file with percentile data. First column should be ages, subsequent columns should be percentiles."
)

if uploaded_file is not None:
    try:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            # Load data
            df = pd.read_csv(uploaded_file, sep='\t', decimal=',')
            
            # Extract percentiles and ages
            percentiles = [int(float(x.replace(',', '.'))) for x in df.columns[1:]]
            ages = df.iloc[:, 0].astype(str).tolist()
            values_matrix = df.iloc[:, 1:].values.astype(float)
            cum_probs = np.array(percentiles) / 100.0
            
            st.success(f"✅ Data loaded successfully! Found {len(ages)} age groups and {len(percentiles)} percentiles.")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
        # Data processing functions
        def detect_and_fix_inverted_data(values_matrix, cum_probs):
            inverted_count = 0
            total_checks = 0
            
            for row in values_matrix:
                if len(row) >= 2:
                    first_val = row[0]
                    last_val = row[-1]
                    if first_val > last_val:
                        inverted_count += 1
                    total_checks += 1
            
            is_inverted = inverted_count > total_checks / 2
            
            if is_inverted:
                st.warning("⚠️ Data appears to be inverted (higher percentiles have lower values). Inverting data to correct format.")
                values_matrix_inverted = np.flip(values_matrix, axis=1)
                return values_matrix_inverted, True
            else:
                return values_matrix, False

        def fit_lognorm_percentiles(values, cum_probs):
            def quantile_fit_loss(params):
                shape, scale = params
                predicted = lognorm.ppf(cum_probs, shape, loc=0, scale=scale)
                return np.sum((predicted - values) ** 2)
            init_shape, _, init_scale = lognorm.fit(values, floc=0)
            res = minimize(quantile_fit_loss, x0=[init_shape, init_scale], bounds=[(0.01, 5), (0.1, 20)])
            shape_opt, scale_opt = res.x
            return shape_opt, scale_opt

        # Process data
        with st.spinner("Processing data and fitting distributions..."):
            # Detect and fix inverted data
            values_matrix, data_was_inverted = detect_and_fix_inverted_data(values_matrix, cum_probs)
            
            # Fit distributions
            fit_results = []
            fit_methods = []
            fit_errors = []
            
            for i, row in enumerate(values_matrix):
                if np.any(row <= 0):
                    st.info(f"ℹ️ Age {ages[i]} contains zero or negative values. Using linear interpolation.")
                    shape, scale = np.nan, np.nan
                    fit_method = 'linear_interp'
                    error = np.nan
                else:
                    shape, scale = fit_lognorm_percentiles(row, cum_probs)
                    predicted = lognorm.ppf(cum_probs, shape, loc=0, scale=scale)
                    error = np.sum((predicted - row) ** 2) / np.sum(row ** 2)
                    if error > ERROR_THRESHOLD:
                        fit_method = 'linear_interp'
                    else:
                        fit_method = 'lognorm'
                
                fit_results.append((ages[i], shape, scale, 0, data_was_inverted, fit_method))
                fit_methods.append(fit_method)
                fit_errors.append(error)
            
            # Create results dataframe
            output_df = pd.DataFrame(fit_results, columns=['age', 'shape', 'scale', 'loc', 'invert_data', 'fit_method'])
            
        # Display results
        st.subheader("📈 Fitting Results")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Age Groups", len(ages))
        with col2:
            lognorm_count = sum(1 for method in fit_methods if method == 'lognorm')
            st.metric("Log-Normal Fits", lognorm_count)
        with col3:
            linear_count = sum(1 for method in fit_methods if method == 'linear_interp')
            st.metric("Linear Interpolations", linear_count)
        
        # Results table
        st.dataframe(output_df, use_container_width=True)
        
        # Download results
        csv_data = output_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"fit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Visualization
        st.subheader("📊 Visualizations")
        
        # Age selector for detailed view
        selected_age_idx = st.selectbox(
            "Select age group to view detailed plots:",
            options=list(range(len(ages))),
            format_func=lambda x: f"Age {ages[x]} ({fit_methods[x]})"
        )
        
        # Create plots for selected age
        def create_age_plots(age_idx):
            vals = values_matrix[age_idx]
            shape, scale, loc, invert_flag, fit_method = fit_results[age_idx][1:]
            percentiles_smooth = np.linspace(1, 99, 500)
            cum_probs_smooth = percentiles_smooth / 100
            
            fig, (ax1, ax2, ax4) = plt.subplots(1, 3, figsize=(18, 5))
            
            # Q-Q plot
            if fit_method == 'lognorm':
                fitted_values = lognorm.ppf(cum_probs_smooth, shape, loc=0, scale=scale)
                ax1.plot(percentiles_smooth, fitted_values, label='Fitted Log-Normal', color='blue')
            else:
                interp_func = interp1d(percentiles, vals, kind='linear', fill_value='extrapolate')
                fitted_values = interp_func(percentiles_smooth)
                ax1.plot(percentiles_smooth, fitted_values, label='Linear Interpolation', color='orange')
            
            ax1.scatter(percentiles, vals, color='red', label='Empirical Data', zorder=5)
            ax1.set_title(f'Value vs Percentile (Age {ages[age_idx]})')
            ax1.set_xlabel('Percentile')
            ax1.set_ylabel('Value')
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
            ax2.set_title('Probability Density')
            ax2.set_xlabel('Value')
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
            ax4.set_title('Cumulative Distribution')
            ax4.set_xlabel('Value')
            ax4.set_ylabel('Cumulative Probability')
            ax4.grid(True)
            ax4.legend()
            
            # Add note about data inversion or fit method
            if invert_flag:
                fit_note = f'(Data was inverted, {fit_method})'
            else:
                fit_note = f'({fit_method})'
            fig.suptitle(f'Age: {ages[age_idx]} {fit_note}', fontsize=16)
            plt.tight_layout()
            return fig
        
        # Display plots
        fig = create_age_plots(selected_age_idx)
        st.pyplot(fig)
        
        # Download all plots
        if st.button("📥 Download All Plots"):
            with st.spinner("Generating all plots..."):
                # Create a zip file with all plots
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i, age in enumerate(ages):
                        fig = create_age_plots(i)
                        img_buffer = BytesIO()
                        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                        img_buffer.seek(0)
                        zip_file.writestr(f"age_{age}_fit.png", img_buffer.getvalue())
                        plt.close(fig)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download All Plots (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"all_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        # JSON Export Section
        st.subheader("📋 JSON Export")
        st.markdown("Copy the JSON data below to use in other systems:")
        
        # Determine the method based on the data
        if all(method == 'lognorm' for method in fit_methods):
            json_method = "lognorm"
        elif all(method == 'linear_interp' for method in fit_methods):
            json_method = "lerp"
        else:
            json_method = "mixed"
        
        # Create JSON data
        if json_method == "lognorm":
            # Log-normal format
            json_data = {
                "data": [
                    {
                        "age": float(age),
                        "loc": float(loc),
                        "scale": float(scale) if not np.isnan(scale) else 0.0,
                        "shape": float(shape) if not np.isnan(shape) else 0.0,
                        "fit_method": "lognorm",
                        "invert_data": bool(invert_flag)
                    }
                    for age, shape, scale, loc, invert_flag, fit_method in fit_results
                ],
                "method": "lognorm"
            }
        else:
            # Linear interpolation format
            json_data = {
                "data": []
            }
            
            for i, (age, shape, scale, loc, invert_flag, fit_method) in enumerate(fit_results):
                age_data = {"age": float(age)}
                
                # Add percentile values
                for j, percentile in enumerate(percentiles):
                    age_data[str(percentile)] = float(values_matrix[i][j])
                
                json_data["data"].append(age_data)
            
            json_data["method"] = "lerp"
        
        # Display JSON with syntax highlighting
        import json
        json_str = json.dumps(json_data, indent=2)
        
        # Create tabs for different formats
        tab1, tab2 = st.tabs(["📄 Formatted JSON", "📋 Raw JSON"])
        
        with tab1:
            st.json(json_data)
        
        with tab2:
            st.code(json_str, language="json")
        
        # Copy to clipboard functionality
        st.markdown("**Copy to clipboard:**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.text_area(
                "JSON Data",
                value=json_str,
                height=400,
                help="Select all text and copy to clipboard"
            )
        
        with col2:
            st.markdown("""
            **Instructions:**
            1. Click in the text area
            2. Press Ctrl+A (Cmd+A on Mac) to select all
            3. Press Ctrl+C (Cmd+C on Mac) to copy
            4. Paste into your target system
            """)
        
        # Download JSON file
        st.download_button(
            label="📥 Download JSON File",
            data=json_str,
            file_name=f"fit_results_{json_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check that your CSV file has the correct format: first column for ages, subsequent columns for percentiles.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Kids2Sport • Log-Normal Distribution Fitter</p>
</div>
""", unsafe_allow_html=True) 
# Log-Normal Distribution Fitter

A Streamlit web application for fitting log-normal distributions to percentile data across different age groups.

## Features

- 📊 **File Upload**: Upload CSV files with percentile data
- 🔄 **Automatic Data Detection**: Detects and handles inverted data
- 📈 **Smart Fitting**: Uses log-normal distribution when appropriate, falls back to linear interpolation
- 📉 **Visualization**: Interactive plots showing Q-Q, PDF, and CDF
- 📥 **Download Results**: Export CSV results and plot images
- ⚙️ **Configurable**: Adjustable error threshold for fitting decisions

## Input Format

Your CSV file should have:
- **First column**: Age groups (e.g., 5.0, 6.0, 7.0, ...)
- **Subsequent columns**: Percentile values (e.g., 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)
- **Decimal separator**: Comma (,) - the app handles this automatically

Example:
```
Age	1	5	10	20	30	40	50	60	70	80	90	95	99
5,0	0,6	0,8	0,9	1,2	1,4	1,7	2,0	2,4	2,9	3,4	4,2	5,0	6,4
6,0	0,6	0,8	0,9	1,2	1,4	1,7	2,0	2,4	2,9	3,4	4,2	5,0	6,4
```

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## Deployment to Streamlit Cloud

### Option 1: Streamlit Cloud (Recommended)

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the path to your app: `streamlit_app.py`
   - Click "Deploy"

### Option 2: Other Platforms

#### Heroku
1. Create a `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Deploy to Heroku using their CLI or GitHub integration.

#### Railway
1. Connect your GitHub repository
2. Railway will automatically detect it's a Python app
3. Deploy with one click

## Usage

1. **Upload your CSV file** using the file uploader
2. **Adjust the error threshold** in the sidebar if needed
3. **View the results** in the data table
4. **Download the results** as a CSV file
5. **Explore visualizations** by selecting different age groups
6. **Download all plots** as a ZIP file

## Output

The app provides:
- **CSV file** with fitting parameters (age, shape, scale, loc, invert_data, fit_method)
- **Individual plots** for each age group showing Q-Q, PDF, and CDF
- **Summary statistics** of the fitting process

## Troubleshooting

### Common Issues

1. **"No module named 'numpy'"**: Install dependencies with `pip install -r requirements.txt`

2. **File format errors**: Ensure your CSV uses tab separation and comma decimal separators

3. **Memory issues**: For large datasets, try processing smaller batches

### Support

If you encounter issues:
1. Check the file format matches the example above
2. Ensure all dependencies are installed
3. Try with a smaller dataset first

## License

This project is open source and available under the MIT License. 
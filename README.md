# Calories Burned Prediction - Linear Regression Implementation

## Project Structure

```
Master-DataMining/
├── data_preprocessing.py
├── linear_regression.py
├── gym_members_exercise_tracking.csv
├── models/
│   └── Linear_Regression.pkl
├── results/
│   └── model_metrics.csv
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Linear Regression Training

```bash
python linear_regression.py
```

This will:
- Load and preprocess the gym dataset
- Train the Linear Regression model using Normal Equation
- Evaluate on both training and test sets
- Calculate all required metrics
- Save the trained model to `models/Linear_Regression.pkl`
- Export results to CSV files in `results/`

## Implementation Details

### Linear Regression Algorithm

The implementation uses the **Normal Equation** method to find optimal weights:

```
θ = (X^T X)^(-1) X^T y
```

Where:
- θ = model parameters (weights + bias)
- X = feature matrix
- y = target values

### Why Normal Equation?

1. **Exact Solution**: Computes the optimal weights directly (no iterations)
2. **Fast**: For datasets with moderate features (< 10,000), it's very efficient
3. **No Hyperparameters**: No learning rate or iterations to tune
4. **Stable**: Guaranteed to converge to the global minimum

### Evaluation Metrics

1. **RMSE (Root Mean Squared Error)**
   - Penalizes larger errors more heavily
   - Interpretable in original units (calories)
   - Formula: `sqrt(mean((y_true - y_pred)^2))`

2. **MAE (Mean Absolute Error)**
   - Robust to outliers
   - Easier to interpret as average prediction error
   - Formula: `mean(|y_true - y_pred|)`

3. **R² (R-squared)**
   - Explains variance captured by the model
   - Range: 0 to 1 (1 = perfect fit)
   - Formula: `1 - (SS_res / SS_tot)`

4. **MAPE (Mean Absolute Percentage Error)**
   - Shows relative error as percentage
   - Useful for understanding prediction quality
   - Formula: `mean(|y_true - y_pred| / y_true) * 100`

5. **Adjusted R²**
   - Accounts for number of features
   - Better for comparing models with different complexities
   - Formula: `1 - [(1 - R²) * (n - 1) / (n - p - 1)]`

## License

MIT License - Data Mining Final Project

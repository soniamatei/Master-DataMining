import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path


class LinearRegression:

    def __init__(self):
        self.weights = None
        self.bias = None
        self.n_features = None
        self.training_time = None

    def fit(self, X, y):
        start_time = time.time()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features = n_features

        X_b = np.c_[np.ones((n_samples, 1)), X]

        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        self.bias = theta[0, 0]
        self.weights = theta[1:].flatten()

        self.training_time = time.time() - start_time

        return self

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X)

        return X.dot(self.weights) + self.bias

    def save_model(self, filepath):
        model_data = {
            'weights': self.weights,
            'bias': self.bias,
            'n_features': self.n_features,
            'training_time': self.training_time
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.weights = model_data['weights']
        self.bias = model_data['bias']
        self.n_features = model_data['n_features']
        self.training_time = model_data['training_time']

        print(f"Model loaded from {filepath}")
        return self


def calculate_metrics(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    mae = np.mean(np.abs(y_true - y_pred))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'MSE': mse
    }


def calculate_adjusted_r2(r2, n_samples, n_features):
    if n_samples <= n_features + 1:
        return np.nan

    adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
    return adj_r2


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Linear_Regression"):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Training completed in {model.training_time:.4f} seconds")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, y_pred_train)

    test_metrics = calculate_metrics(y_test, y_pred_test)

    n_samples = len(y_test)
    n_features = X_test.shape[1]
    adjusted_r2 = calculate_adjusted_r2(test_metrics['R2'], n_samples, n_features)

    print(f"\nTraining Set Performance:")
    print(f"  RMSE: {train_metrics['RMSE']:.4f}")
    print(f"  MAE:  {train_metrics['MAE']:.4f}")
    print(f"  R²:   {train_metrics['R2']:.4f}")
    print(f"  MAPE: {train_metrics['MAPE']:.4f}%")

    print(f"\nTest Set Performance:")
    print(f"  RMSE:         {test_metrics['RMSE']:.4f}")
    print(f"  MAE:          {test_metrics['MAE']:.4f}")
    print(f"  R²:           {test_metrics['R2']:.4f}")
    print(f"  MAPE:         {test_metrics['MAPE']:.4f}%")
    print(f"  Adjusted R²:  {adjusted_r2:.4f}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"{model_name}.pkl"
    model.save_model(model_path)

    results = {
        'Model': model_name,
        'Training_Time_seconds': model.training_time,
        'Train_RMSE': train_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Train_R2': train_metrics['R2'],
        'Test_RMSE': test_metrics['RMSE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_R2': test_metrics['R2'],
        'Test_MAPE': test_metrics['MAPE'],
        'Test_Adjusted_R2': adjusted_r2,
        'N_Features': n_features,
        'N_Train_Samples': len(y_train),
        'N_Test_Samples': n_samples
    }

    return model, results


def save_results_to_csv(results, metrics_filename="model_metrics.csv"):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if isinstance(results, dict):
        results = [results]

    df_metrics = pd.DataFrame(results)

    metrics_path = results_dir / metrics_filename
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)

    print("\nEvaluation Metrics:")
    metrics_cols = ['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE', 'Test_Adjusted_R2']
    print(df_metrics[metrics_cols].to_string(index=False))

    print("\nTraining Time:")
    time_cols = ['Model', 'Training_Time_seconds']
    print(df_metrics[time_cols].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    from data_preprocessing import preprocess_gym_data

    print("Loading and preprocessing data...")
    data = preprocess_gym_data('gym_members_exercise_tracking.csv')

    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']

    model, results = train_and_evaluate(X_train, X_test, y_train, y_test)

    save_results_to_csv(results)

    print("\n" + "="*60)
    print("Linear Regression Training Complete!")
    print("="*60)
    print(f"\nModel saved in: models/")
    print(f"Results saved in: results/")

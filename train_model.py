import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from datetime import datetime

# Time series models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import tensorflow as tf
from xgboost import XGBRegressor
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from filterpy.kalman import KalmanFilter
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self, data_path='TSLA.csv'):
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Data preprocessing
        self.scaler = MinMaxScaler()
        self.data['Close_scaled'] = self.scaler.fit_transform(self.data[['Close']])
    
    def save_model(self, model, model_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        path = os.path.join(self.models_dir, filename)
        joblib.dump(model, path)
        return path
    
    def train_arima(self):
        model = ARIMA(self.data['Close'], order=(5,1,0))
        model_fit = model.fit()
        return self.save_model(model_fit, 'ARIMA')
    
    def train_sarima(self):
        model = SARIMAX(self.data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        return self.save_model(model_fit, 'SARIMA')
    
    def train_prophet(self):
        df_prophet = pd.DataFrame({
            'ds': self.data['Date'],
            'y': self.data['Close']
        })
        model = Prophet()
        model.fit(df_prophet)
        return self.save_model(model, 'Prophet')
    
    def train_lstm(self):
        # Simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(60, 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        # Save in TensorFlow format
        path = os.path.join(self.models_dir, f'LSTM_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        model.save(path)
        return path
    
    # ... Other model training methods ...
    
    def train_all_models(self):
        """Train all models and return a list of paths to the trained models"""
        trained_models = []
        
        try:
            print("Training ARIMA...")
            trained_models.append(self.train_arima())
        except Exception as e:
            print(f"ARIMA training failed: {str(e)}")
            
        try:
            print("Training SARIMA...")
            trained_models.append(self.train_sarima())
        except Exception as e:
            print(f"SARIMA training failed: {str(e)}")
            
        try:
            print("Training Prophet...")
            trained_models.append(self.train_prophet())
        except Exception as e:
            print(f"Prophet training failed: {str(e)}")
            
        try:
            print("Training LSTM...")
            trained_models.append(self.train_lstm())
        except Exception as e:
            print(f"LSTM training failed: {str(e)}")
            
        # ... Training for other models ...
        
        return trained_models

    def _prepare_sequence_data(self, data, seq_length):
        """Prepare sequence data for LSTM, GRU, etc."""
        X, y = [], []
        data = np.array(data)
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # For XGBoost and RandomForest, 2D arrays are needed
        if len(X.shape) == 3:  # If it's a 3D array (for LSTM/GRU)
            X_2d = X.reshape(X.shape[0], -1)  # Flatten to 2D
            return {'3d': (X, y), '2d': (X_2d, y)}
        return {'3d': (X, y), '2d': (X, y)}

    def train_single_model(self, model_name, training_data):
        """Train a single model"""
        try:
            # Ensure data is a copy, not a view
            training_data = training_data.copy()
            
            if model_name in ['LSTM', 'GRU', 'XGBoost', 'Random Forest']:
                # Prepare sequence data
                data_dict = self._prepare_sequence_data(training_data['Close'].values, 10)
                
                if model_name == 'LSTM':
                    X, y = data_dict['3d']
                    # Ensure correct input shape
                    X = X.reshape(-1, 10, 1)
                    model = tf.keras.Sequential([
                        tf.keras.layers.LSTM(20, input_shape=(10, 1)),
                        tf.keras.layers.Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                    
                    # Add .keras extension
                    path = os.path.join(self.models_dir, 
                                      f'LSTM_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')
                    model.save(path, save_format='keras')
                    return path
                
                elif model_name == 'GRU':
                    X, y = data_dict['3d']
                    # Ensure correct input shape
                    X = X.reshape(-1, 10, 1)
                    model = tf.keras.Sequential([
                        tf.keras.layers.GRU(20, input_shape=(10, 1)),
                        tf.keras.layers.Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                    
                    # Add .keras extension
                    path = os.path.join(self.models_dir, 
                                      f'GRU_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')
                    model.save(path, save_format='keras')
                    return path
                
                elif model_name == 'XGBoost':
                    X, y = data_dict['2d']  # Use 2D data
                    model = XGBRegressor(
                        n_estimators=10,
                        max_depth=3,
                        learning_rate=0.1,
                        verbosity=0
                    )
                    model.fit(X, y)
                    return self.save_model(model, 'XGBoost')
                
                elif model_name == 'Random Forest':
                    X, y = data_dict['2d']  # Use 2D data
                    model = RandomForestRegressor(
                        n_estimators=10,
                        max_depth=3,
                        n_jobs=-1,
                        verbose=0
                    )
                    model.fit(X, y)
                    return self.save_model(model, 'RandomForest')
            
            elif model_name == 'ARIMA':
                model = ARIMA(training_data['Close'], order=(2,1,0))
                model_fit = model.fit(method='css', maxiter=10, disp=0)  # Reduce output
                return self.save_model(model_fit, 'ARIMA')
            
            elif model_name == 'Prophet':
                df_prophet = pd.DataFrame({
                    'ds': training_data['Date'],
                    'y': training_data['Close']
                })
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    mcmc_samples=0
                )
                model.fit(df_prophet)
                return self.save_model(model, 'Prophet')
            
            elif model_name == 'SARIMA':
                # Simplify SARIMA model parameters
                model = SARIMAX(training_data['Close'], 
                              order=(1,1,1),           # Reduce order
                              seasonal_order=(1,1,0,5)) # Reduce seasonal parameters
                model_fit = model.fit(maxiter=10, method='powell')
                return self.save_model(model_fit, 'SARIMA')
            
            elif model_name == 'VAR':
                # Simplify VAR model
                model = VAR(training_data[['Close', 'Volume']])
                model_fit = model.fit(maxlags=5, ic='aic', trend='c')  # Reduce max lag order
                return self.save_model(model_fit, 'VAR')
            
            elif model_name == 'Exponential Smoothing':
                # Simplify exponential smoothing model
                model = ExponentialSmoothing(
                    training_data['Close'],
                    seasonal_periods=5,     # Reduce seasonal periods
                    trend='add',
                    seasonal='add',
                )
                model_fit = model.fit(optimized=True, use_boxcox=False)
                return self.save_model(model_fit, 'ExpSmoothing')
            
            elif model_name == 'Kalman Filter':
                # Simplify Kalman filter
                kf = KalmanFilter(dim_x=2, dim_z=1)
                kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
                kf.H = np.array([[1., 0.]])            # Measurement function
                kf.R *= 5                              # Measurement noise
                kf.Q *= 0.1                           # Process noise
                return self.save_model(kf, 'KalmanFilter')
            
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
        except Exception as e:
            print(f"Training error details: {str(e)}")  # Add detailed error info
            raise Exception(f"Failed to train model {model_name}: {str(e)}")
    
    def evaluate_predictions(self, predictions, actual_values):
        """Evaluate prediction results"""
        results = {}
        for model_name, pred_values in predictions.items():
            mse = np.mean((pred_values - actual_values) ** 2)
            mae = np.mean(np.abs(pred_values - actual_values))
            rmse = np.sqrt(mse)
            results[model_name] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse
            }
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models()
    print("Trained models saved at:", trained_models)
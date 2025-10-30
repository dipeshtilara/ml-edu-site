# Time Series: Sales Forecasting
# ARIMA-like model for time series prediction

import math
import random
from typing import List, Tuple

class TimeSeriesForecaster:
    """
    Simple time series forecasting using moving average and trend analysis.
    Implements concepts similar to ARIMA but simplified for educational purposes.
    """
    
    def __init__(self, window_size: int = 7, trend_window: int = 14):
        """
        Initialize time series forecaster.
        
        Args:
            window_size: Window for moving average
            trend_window: Window for trend calculation
        """
        self.window_size = window_size
        self.trend_window = trend_window
        self.training_data = None
        self.seasonality_factors = None
        self.trend_coefficient = None
    
    def moving_average(self, data: List[float], window: int) -> List[float]:
        """
        Calculate moving average.
        
        Args:
            data: Time series data
            window: Window size
            
        Returns:
            List of moving averages
        """
        ma = []
        for i in range(len(data)):
            if i < window - 1:
                # Not enough data points, use available
                avg = sum(data[:i+1]) / (i + 1)
            else:
                avg = sum(data[i-window+1:i+1]) / window
            ma.append(avg)
        return ma
    
    def calculate_trend(self, data: List[float]) -> float:
        """
        Calculate linear trend coefficient.
        
        Args:
            data: Time series data
            
        Returns:
            Trend coefficient (slope)
        """
        n = len(data)
        if n < 2:
            return 0.0
        
        # Simple linear regression
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(data) / n
        
        numerator = sum((x[i] - mean_x) * (data[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def detect_seasonality(self, data: List[float], period: int = 7) -> List[float]:
        """
        Detect seasonal patterns.
        
        Args:
            data: Time series data
            period: Seasonal period (e.g., 7 for weekly)
            
        Returns:
            List of seasonal factors
        """
        n = len(data)
        seasonal_factors = [0.0] * period
        counts = [0] * period
        
        # Calculate average for each position in the cycle
        for i in range(n):
            position = i % period
            seasonal_factors[position] += data[i]
            counts[position] += 1
        
        # Normalize
        overall_mean = sum(data) / n
        for i in range(period):
            if counts[i] > 0:
                seasonal_factors[i] = (seasonal_factors[i] / counts[i]) / overall_mean
            else:
                seasonal_factors[i] = 1.0
        
        return seasonal_factors
    
    def remove_seasonality(self, data: List[float], seasonal_factors: List[float], period: int = 7) -> List[float]:
        """
        Remove seasonal component from data.
        
        Args:
            data: Time series data
            seasonal_factors: Seasonal factors
            period: Seasonal period
            
        Returns:
            Deseasonalized data
        """
        deseasonalized = []
        for i in range(len(data)):
            position = i % period
            deseasonalized.append(data[i] / seasonal_factors[position])
        return deseasonalized
    
    def fit(self, data: List[float], seasonal_period: int = 7):
        """
        Fit the time series model.
        
        Args:
            data: Historical time series data
            seasonal_period: Length of seasonal cycle
        """
        self.training_data = data[:]
        
        # Detect seasonality
        self.seasonality_factors = self.detect_seasonality(data, seasonal_period)
        
        # Remove seasonality to calculate trend
        deseasonalized = self.remove_seasonality(data, self.seasonality_factors, seasonal_period)
        
        # Calculate trend on recent data
        recent_data = deseasonalized[-self.trend_window:]
        self.trend_coefficient = self.calculate_trend(recent_data)
    
    def forecast(self, steps: int, seasonal_period: int = 7) -> List[float]:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            seasonal_period: Length of seasonal cycle
            
        Returns:
            List of forecasted values
        """
        if not self.training_data:
            return []
        
        forecasts = []
        last_value = self.training_data[-1]
        base_index = len(self.training_data)
        
        # Get moving average of recent data
        recent_ma = self.moving_average(self.training_data, self.window_size)
        base_level = recent_ma[-1]
        
        for i in range(steps):
            # Apply trend
            trend_component = self.trend_coefficient * i
            
            # Apply seasonality
            position = (base_index + i) % seasonal_period
            seasonal_component = self.seasonality_factors[position]
            
            # Combine components
            forecast = (base_level + trend_component) * seasonal_component
            forecasts.append(forecast)
        
        return forecasts
    
    def evaluate(self, test_data: List[float], seasonal_period: int = 7) -> dict:
        """
        Evaluate forecast accuracy.
        
        Args:
            test_data: Actual future values
            seasonal_period: Length of seasonal cycle
            
        Returns:
            Dictionary of evaluation metrics
        """
        forecasts = self.forecast(len(test_data), seasonal_period)
        
        # Calculate metrics
        errors = [actual - pred for actual, pred in zip(test_data, forecasts)]
        abs_errors = [abs(e) for e in errors]
        squared_errors = [e**2 for e in errors]
        
        mae = sum(abs_errors) / len(abs_errors)
        mse = sum(squared_errors) / len(squared_errors)
        rmse = math.sqrt(mse)
        
        # Mean Absolute Percentage Error
        mape = sum(abs(e) / actual * 100 for e, actual in zip(errors, test_data) if actual != 0) / len(test_data)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'forecasts': forecasts,
            'actuals': test_data
        }


def generate_sales_data(n_days: int = 180) -> Tuple[List[float], List[float]]:
    """
    Generate synthetic sales data with trend and seasonality.
    
    Args:
        n_days: Number of days to generate
        
    Returns:
        Tuple of (training_data, test_data)
    """
    data = []
    base_sales = 100.0
    trend = 0.5  # Growing trend
    
    # Weekly seasonality (higher on weekends)
    weekly_pattern = [0.8, 0.85, 0.9, 0.95, 1.0, 1.3, 1.4]
    
    for day in range(n_days):
        # Base with trend
        value = base_sales + trend * day
        
        # Add weekly seasonality
        day_of_week = day % 7
        value *= weekly_pattern[day_of_week]
        
        # Add monthly seasonality (peak mid-month)
        day_of_month = day % 30
        if 10 <= day_of_month <= 20:
            value *= 1.2
        
        # Add noise
        noise = random.gauss(0, 10)
        value += noise
        
        data.append(max(0, value))
    
    # Split into train/test (80/20)
    split = int(0.8 * n_days)
    return data[:split], data[split:]


if __name__ == "__main__":
    print("Time Series: Sales Forecasting")
    print("="*60)
    
    # Generate sales data
    print("\nGenerating sales time series data...")
    train_data, test_data = generate_sales_data(n_days=180)
    print(f"Training period: {len(train_data)} days")
    print(f"Test period: {len(test_data)} days")
    
    # Fit model
    print("\nFitting time series model...")
    forecaster = TimeSeriesForecaster(window_size=7, trend_window=14)
    forecaster.fit(train_data, seasonal_period=7)
    
    print(f"\nDetected trend coefficient: {forecaster.trend_coefficient:.4f}")
    print(f"Seasonal factors (weekly):")
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, (day, factor) in enumerate(zip(days, forecaster.seasonality_factors)):
        print(f"  {day}: {factor:.3f}")
    
    # Evaluate
    print("\nForecasting next 36 days...")
    results = forecaster.evaluate(test_data, seasonal_period=7)
    
    print(f"\nForecast Accuracy:")
    print(f"  MAE: {results['mae']:.2f}")
    print(f"  RMSE: {results['rmse']:.2f}")
    print(f"  MAPE: {results['mape']:.2f}%")
    
    print("\n" + "="*60)
    print("Time Series Forecasting Complete!")

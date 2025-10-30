# Random Forest: Stock Price Prediction
# CBSE Class 12 AI - Ensemble Learning Project
# This project implements Random Forest from scratch for stock market analysis

import math
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import statistics

def print_header():
    """Print project header with information"""
    print("=" * 80)
    print("RANDOM FOREST: STOCK PRICE PREDICTION")
    print("CBSE Class 12 AI - Ensemble Learning Project")
    print("=" * 80)
    print()

class TreeNode:
    """Node class for decision tree structure in Random Forest"""
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None, samples=0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For regression: mean target value
        self.samples = samples
        self.variance = 0.0
        self.mse_reduction = 0.0

class DecisionTreeRegressor:
    """Decision Tree for Regression (used within Random Forest)"""
    
    def __init__(self, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # Number of features to consider at each split
        self.root = None
        self.feature_indices = None  # Random subset of features for this tree
    
    def calculate_mse(self, y: List[float]) -> float:
        """Calculate Mean Squared Error (variance for regression)"""
        if not y:
            return 0.0
        
        mean_y = sum(y) / len(y)
        mse = sum((yi - mean_y) ** 2 for yi in y) / len(y)
        return mse
    
    def find_best_split(self, X: List[List[float]], y: List[float], feature_subset: List[int]) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best feature and threshold to split on from feature subset"""
        n_samples = len(X)
        
        if n_samples <= 1:
            return None, None, 0.0
        
        # Calculate parent MSE
        parent_mse = self.calculate_mse(y)
        best_reduction = 0.0
        best_feature = None
        best_threshold = None
        
        # Try each feature in the random subset
        for feature_idx in feature_subset:
            # Get unique values for this feature
            feature_values = sorted(set(X[i][feature_idx] for i in range(n_samples)))
            
            # Try each unique value as a threshold
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                
                # Split data
                left_y, right_y = [], []
                
                for j in range(n_samples):
                    if X[j][feature_idx] <= threshold:
                        left_y.append(y[j])
                    else:
                        right_y.append(y[j])
                
                # Skip if split doesn't meet minimum sample requirements
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate MSE reduction
                left_mse = self.calculate_mse(left_y)
                right_mse = self.calculate_mse(right_y)
                
                weighted_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / n_samples
                mse_reduction = parent_mse - weighted_mse
                
                # Update best split if this is better
                if mse_reduction > best_reduction:
                    best_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def build_tree(self, X: List[List[float]], y: List[float], depth: int = 0) -> TreeNode:
        """Recursively build the regression tree"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Create node
        node = TreeNode(samples=n_samples)
        node.variance = self.calculate_mse(y)
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            node.variance < 1e-6):
            
            # Make this a leaf node with mean prediction
            node.value = sum(y) / len(y) if y else 0.0
            return node
        
        # Select random subset of features
        if self.max_features is None:
            feature_subset = list(range(n_features))
        else:
            max_feat = min(self.max_features, n_features)
            feature_subset = random.sample(range(n_features), max_feat)
        
        # Find best split
        best_feature, best_threshold, best_reduction = self.find_best_split(X, y, feature_subset)
        
        if best_feature is None or best_reduction <= 0:
            # Make this a leaf node
            node.value = sum(y) / len(y) if y else 0.0
            return node
        
        # Set split parameters
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.mse_reduction = best_reduction
        
        # Split data
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i in range(n_samples):
            if X[i][best_feature] <= best_threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        # Recursively build subtrees
        node.left = self.build_tree(left_X, left_y, depth + 1)
        node.right = self.build_tree(right_X, right_y, depth + 1)
        
        return node
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Train the decision tree regressor"""
        n_features = len(X[0]) if X else 0
        
        # Set default max_features if not specified
        if self.max_features is None:
            self.max_features = max(1, int(math.sqrt(n_features)))
        
        self.root = self.build_tree(X, y)
    
    def predict_sample(self, sample: List[float], node: TreeNode = None) -> float:
        """Predict value for a single sample"""
        if node is None:
            node = self.root
        
        # If leaf node, return the predicted value
        if node.value is not None:
            return node.value
        
        # Navigate to appropriate child
        if sample[node.feature_index] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions for multiple samples"""
        if self.root is None:
            raise ValueError("Tree has not been trained yet. Call fit() first.")
        
        return [self.predict_sample(sample) for sample in X]

class RandomForestRegressor:
    """Complete Random Forest implementation for regression"""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=5, 
                 min_samples_leaf=2, max_features=None, bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_importances = None
        self.oob_score = None
        
        if random_state is not None:
            random.seed(random_state)
    
    def bootstrap_sample(self, X: List[List[float]], y: List[float]) -> Tuple[List[List[float]], List[float], List[int]]:
        """Create bootstrap sample with replacement"""
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        
        X_bootstrap = [X[i] for i in indices]
        y_bootstrap = [y[i] for i in indices]
        
        # Track out-of-bag samples
        oob_indices = [i for i in range(n_samples) if i not in set(indices)]
        
        return X_bootstrap, y_bootstrap, oob_indices
    
    def calculate_feature_importance(self, X: List[List[float]]) -> List[float]:
        """Calculate feature importance based on MSE reduction"""
        n_features = len(X[0]) if X else 0
        feature_importances = [0.0] * n_features
        
        def traverse_tree(node, total_samples):
            if node is None or node.value is not None:
                return
            
            # Add importance based on weighted MSE reduction
            importance = (node.samples / total_samples) * node.mse_reduction
            feature_importances[node.feature_index] += importance
            
            # Traverse children
            traverse_tree(node.left, total_samples)
            traverse_tree(node.right, total_samples)
        
        # Sum importance from all trees
        for tree in self.trees:
            if tree.root:
                traverse_tree(tree.root, len(X))
        
        # Normalize
        total_importance = sum(feature_importances)
        if total_importance > 0:
            feature_importances = [imp / total_importance for imp in feature_importances]
        
        return feature_importances
    
    def fit(self, X: List[List[float]], y: List[float]):
        """Train the Random Forest"""
        print(f"Training Random Forest with {self.n_estimators} trees...")
        
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # Set default max_features if not specified
        if self.max_features is None:
            self.max_features = max(1, n_features // 3)
        
        self.trees = []
        all_oob_predictions = [[] for _ in range(n_samples)]
        
        # Train each tree
        for i in range(self.n_estimators):
            if (i + 1) % 20 == 0:
                print(f"Training tree {i + 1}/{self.n_estimators}...")
            
            # Create decision tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            
            if self.bootstrap:
                # Bootstrap sampling
                X_bootstrap, y_bootstrap, oob_indices = self.bootstrap_sample(X, y)
                
                # Train tree on bootstrap sample
                tree.fit(X_bootstrap, y_bootstrap)
                
                # Make predictions on out-of-bag samples
                if oob_indices:
                    X_oob = [X[idx] for idx in oob_indices]
                    oob_predictions = tree.predict(X_oob)
                    
                    for j, idx in enumerate(oob_indices):
                        all_oob_predictions[idx].append(oob_predictions[j])
            else:
                # Use full dataset
                tree.fit(X, y)
            
            self.trees.append(tree)
        
        # Calculate out-of-bag score
        if self.bootstrap:
            oob_targets = []
            oob_predictions = []
            
            for i in range(n_samples):
                if all_oob_predictions[i]:  # If sample was out-of-bag for some trees
                    oob_targets.append(y[i])
                    oob_predictions.append(statistics.mean(all_oob_predictions[i]))
            
            if oob_predictions:
                mse = sum((oob_targets[i] - oob_predictions[i]) ** 2 for i in range(len(oob_predictions))) / len(oob_predictions)
                self.oob_score = mse
        
        # Calculate feature importances
        self.feature_importances = self.calculate_feature_importance(X)
        
        print(f"Random Forest training completed!")
        if self.oob_score is not None:
            print(f"Out-of-bag MSE: {self.oob_score:.4f}")
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions using ensemble of trees"""
        if not self.trees:
            raise ValueError("Forest has not been trained yet. Call fit() first.")
        
        # Get predictions from all trees
        all_predictions = []
        for tree in self.trees:
            tree_predictions = tree.predict(X)
            all_predictions.append(tree_predictions)
        
        # Average predictions across all trees
        ensemble_predictions = []
        for i in range(len(X)):
            tree_preds = [all_predictions[j][i] for j in range(len(self.trees))]
            ensemble_pred = statistics.mean(tree_preds)
            ensemble_predictions.append(ensemble_pred)
        
        return ensemble_predictions
    
    def get_feature_importance(self) -> List[float]:
        """Get feature importance scores"""
        return self.feature_importances.copy() if self.feature_importances else []

class StockDataGenerator:
    """Generate realistic stock market dataset"""
    
    @staticmethod
    def generate_stock_data(n_days: int = 1000) -> Tuple[List[List[float]], List[float], List[str]]:
        """Generate synthetic stock market dataset with technical indicators"""
        random.seed(42)  # For reproducible results
        
        feature_names = [
            'open_price', 'high_price', 'low_price', 'volume', 
            'sma_5', 'sma_20', 'rsi', 'macd', 'bollinger_upper', 
            'bollinger_lower', 'volatility', 'price_change_1d', 
            'price_change_5d', 'volume_ratio'
        ]
        
        # Initialize with starting values
        base_price = 100.0
        prices = [base_price]
        volumes = []
        
        # Generate price series with realistic market behavior
        for day in range(n_days):
            # Random walk with trend and volatility
            trend = 0.0001  # Slight upward trend
            volatility = 0.02
            
            # Add market cycles and events
            if day > 200:
                # Bull market period
                trend += 0.0005
            if 400 <= day <= 450:
                # Market correction
                trend -= 0.003
                volatility *= 2
            if day > 700:
                # Recovery period
                trend += 0.0008
                volatility *= 0.7
            
            # Random price change
            price_change = random.gauss(trend, volatility)
            new_price = prices[-1] * (1 + price_change)
            new_price = max(10, new_price)  # Prevent negative prices
            prices.append(new_price)
            
            # Generate volume (inversely correlated with price stability)
            base_volume = 1000000
            volume_volatility = abs(price_change) * 5000000
            volume = max(100000, int(random.gauss(base_volume + volume_volatility, base_volume * 0.3)))
            volumes.append(volume)
        
        # Calculate technical indicators
        stock_features = []
        target_prices = []
        
        for i in range(20, len(prices) - 1):  # Start from day 20 to have enough history
            current_price = prices[i]
            next_price = prices[i + 1]  # Target: next day's closing price
            
            # Basic OHLC (simplified: assume close = current_price)
            daily_volatility = random.uniform(0.01, 0.03)
            open_price = current_price * random.uniform(0.995, 1.005)
            high_price = current_price * (1 + daily_volatility * random.uniform(0.5, 1.5))
            low_price = current_price * (1 - daily_volatility * random.uniform(0.5, 1.5))
            
            # Simple Moving Averages
            sma_5 = sum(prices[i-4:i+1]) / 5
            sma_20 = sum(prices[i-19:i+1]) / 20
            
            # RSI (Relative Strength Index) - simplified
            price_changes = [prices[j] - prices[j-1] for j in range(i-13, i+1)]
            gains = [change if change > 0 else 0 for change in price_changes]
            losses = [-change if change < 0 else 0 for change in price_changes]
            
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # MACD (simplified)
            ema_12 = sum(prices[i-11:i+1]) / 12  # Simplified EMA as SMA
            ema_26 = sum(prices[i-25:i+1]) / 26
            macd = ema_12 - ema_26
            
            # Bollinger Bands
            sma_20_list = prices[i-19:i+1]
            bb_std = (sum([(p - sma_20) ** 2 for p in sma_20_list]) / 20) ** 0.5
            bollinger_upper = sma_20 + (2 * bb_std)
            bollinger_lower = sma_20 - (2 * bb_std)
            
            # Volatility (standard deviation of last 10 days)
            recent_prices = prices[i-9:i+1]
            volatility = (sum([(p - sum(recent_prices)/10) ** 2 for p in recent_prices]) / 10) ** 0.5
            
            # Price changes
            price_change_1d = (current_price - prices[i-1]) / prices[i-1]
            price_change_5d = (current_price - prices[i-5]) / prices[i-5]
            
            # Volume ratio (current volume vs average)
            current_volume = volumes[i-20]  # Adjust index for volumes
            avg_volume = sum(volumes[i-25:i-20]) / 5
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            features = [
                open_price, high_price, low_price, current_volume,
                sma_5, sma_20, rsi, macd, bollinger_upper, bollinger_lower,
                volatility, price_change_1d, price_change_5d, volume_ratio
            ]
            
            stock_features.append(features)
            target_prices.append(next_price)
        
        return stock_features, target_prices, feature_names

class RegressionEvaluator:
    """Comprehensive evaluation metrics for regression"""
    
    @staticmethod
    def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Mean Squared Error"""
        return sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))]) / len(y_true)
    
    @staticmethod
    def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        return sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]) / len(y_true)
    
    @staticmethod
    def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Root Mean Squared Error"""
        mse = RegressionEvaluator.mean_squared_error(y_true, y_pred)
        return math.sqrt(mse)
    
    @staticmethod
    def r_squared(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate R-squared (coefficient of determination)"""
        y_mean = sum(y_true) / len(y_true)
        ss_tot = sum([(y - y_mean) ** 2 for y in y_true])
        ss_res = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: List[float], y_pred: List[float]) -> float:
        """Calculate Mean Absolute Percentage Error"""
        if not y_true or any(y == 0 for y in y_true):
            return float('inf')
        
        mape = sum([abs((y_true[i] - y_pred[i]) / y_true[i]) for i in range(len(y_true))]) / len(y_true)
        return mape * 100  # Convert to percentage

def split_data(X: List[List[float]], y: List[float], train_ratio: float = 0.8) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """Split data into training and testing sets maintaining temporal order for time series"""
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    # For time series, maintain temporal order
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    return X_train, X_test, y_train, y_test

def print_data_analysis(X: List[List[float]], y: List[float], feature_names: List[str]):
    """Print comprehensive data analysis"""
    print("\n" + "=" * 60)
    print("STOCK MARKET DATASET ANALYSIS")
    print("=" * 60)
    
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    print(f"Number of trading days: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Technical indicators: {', '.join(feature_names[:5])}...")
    
    # Target variable statistics
    y_mean = sum(y) / len(y)
    y_min = min(y)
    y_max = max(y)
    print(f"\nStock Price Statistics:")
    print(f"Average price: ${y_mean:.2f}")
    print(f"Price range: ${y_min:.2f} - ${y_max:.2f}")
    
    # Feature statistics (show first few features)
    print(f"\nKey Feature Statistics:")
    print(f"{'Feature':<20} {'Mean':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    
    for j in range(min(8, n_features)):  # Show first 8 features
        feature_values = [X[i][j] for i in range(n_samples)]
        mean_val = sum(feature_values) / len(feature_values)
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        print(f"{feature_names[j]:<20} {mean_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f}")

def print_evaluation_results(y_true: List[float], y_pred: List[float], model: RandomForestRegressor, feature_names: List[str]):
    """Print comprehensive evaluation results"""
    print("\n" + "=" * 60)
    print("RANDOM FOREST EVALUATION RESULTS")
    print("=" * 60)
    
    evaluator = RegressionEvaluator()
    
    # Calculate metrics
    mse = evaluator.mean_squared_error(y_true, y_pred)
    mae = evaluator.mean_absolute_error(y_true, y_pred)
    rmse = evaluator.root_mean_squared_error(y_true, y_pred)
    r2 = evaluator.r_squared(y_true, y_pred)
    mape = evaluator.mean_absolute_percentage_error(y_true, y_pred)
    
    print(f"\nPrediction Accuracy:")
    print(f"Mean Squared Error (MSE):      ${mse:.4f}")
    print(f"Mean Absolute Error (MAE):     ${mae:.4f}")
    print(f"Root Mean Squared Error:       ${rmse:.4f}")
    print(f"R-squared (R²):               {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Model information
    print(f"\nModel Configuration:")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max tree depth: {model.max_depth}")
    print(f"Features per split: {model.max_features}")
    
    if model.oob_score is not None:
        print(f"Out-of-bag MSE: {model.oob_score:.4f}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance:
        print(f"\nTop 10 Most Important Features:")
        print(f"{'Feature':<20} {'Importance':<12}")
        print("-" * 35)
        
        # Sort by importance
        importance_pairs = [(feature_names[i], feature_importance[i]) for i in range(len(feature_names))]
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importance_pairs[:10]):
            print(f"{feature:<20} {importance:<12.4f}")
    
    # Sample predictions
    print(f"\nSample Predictions (Last 10 test cases):")
    print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
    print("-" * 50)
    
    for i in range(max(0, len(y_true)-10), len(y_true)):
        actual = y_true[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        error_pct = (error / actual * 100) if actual != 0 else 0
        
        print(f"${actual:<11.2f} ${predicted:<11.2f} ${error:<11.2f} {error_pct:<9.2f}%")

def main():
    """Main function to demonstrate Random Forest for stock prediction"""
    print_header()
    
    # Generate stock market dataset
    print("Generating stock market dataset with technical indicators...")
    X, y, feature_names = StockDataGenerator.generate_stock_data(n_days=1000)
    
    # Analyze dataset
    print_data_analysis(X, y, feature_names)
    
    # Split data (maintaining temporal order)
    print("\nSplitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)
    print(f"Training set: {len(X_train)} days")
    print(f"Testing set: {len(X_test)} days")
    
    # Create and train Random Forest
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)
    
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=5,
        bootstrap=True,
        random_state=42
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model
    print_evaluation_results(y_test, y_pred, rf_model, feature_names)
    
    print("\n" + "=" * 60)
    print("STOCK PREDICTION PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': rf_model,
        'test_data': (X_test, y_test),
        'predictions': y_pred,
        'feature_names': feature_names,
        'metrics': {
            'mse': RegressionEvaluator.mean_squared_error(y_test, y_pred),
            'r2': RegressionEvaluator.r_squared(y_test, y_pred)
        }
    }

if __name__ == "__main__":
    results = main()

# Dependencies and Notes:
# This project implements Random Forest completely from scratch using only Python built-ins.
# 
# Key Dependencies:
# - math: For mathematical operations like sqrt(), log()
# - random: For bootstrap sampling, feature selection, and data generation
# - typing: For type hints (List, Tuple, Dict, Any, Optional)
# - collections.Counter: For counting class occurrences in trees
# - statistics: For calculating mean in ensemble predictions
# 
# Educational Notes:
# 1. Random Forest combines multiple decision trees through bootstrap aggregation
# 2. Each tree uses a random subset of features for splitting (feature bagging)
# 3. Bootstrap sampling creates diverse training sets for each tree
# 4. Out-of-bag (OOB) samples provide unbiased performance estimates
# 5. Feature importance is calculated based on MSE reduction across all trees
# 6. Ensemble methods typically achieve better performance than single models
#
# This implementation demonstrates:
# - Complete Random Forest algorithm with bootstrap aggregation
# - Financial time series prediction with technical indicators
# - Feature importance analysis for model interpretability
# - Comprehensive evaluation metrics for regression problems
# - Realistic stock market data generation with technical analysis
# - Out-of-bag error estimation for model validation
# - Ensemble learning principles and their advantages over single models
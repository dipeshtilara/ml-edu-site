# K-Means Clustering: Customer Segmentation Analysis
# CBSE Class 12 AI - Unsupervised Learning Project
# This project implements K-Means clustering from scratch for customer analysis

import math
import random
from typing import List, Tuple, Dict, Any, Optional

def print_header():
    """Print project header with information"""
    print("=" * 80)
    print("K-MEANS CLUSTERING: CUSTOMER SEGMENTATION ANALYSIS")
    print("CBSE Class 12 AI - Unsupervised Learning Project")
    print("=" * 80)
    print()

class KMeansClusterer:
    """Complete K-Means clustering implementation from scratch"""
    
    def __init__(self, k=3, max_iters=100, tolerance=1e-4, init_method='random'):
        self.k = k  # Number of clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.init_method = init_method
        
        # Model state
        self.centroids = None
        self.labels = None
        self.cluster_assignments = None
        self.inertia_history = []
        self.converged = False
        self.n_iterations = 0
    
    def euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def manhattan_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Manhattan distance between two points"""
        return sum(abs(a - b) for a, b in zip(point1, point2))
    
    def cosine_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Cosine distance between two points"""
        dot_product = sum(a * b for a, b in zip(point1, point2))
        norm1 = math.sqrt(sum(a ** 2 for a in point1))
        norm2 = math.sqrt(sum(b ** 2 for b in point2))
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        cosine_similarity = dot_product / (norm1 * norm2)
        return 1 - cosine_similarity
    
    def initialize_centroids(self, X: List[List[float]]) -> List[List[float]]:
        """Initialize centroids using different methods"""
        n_samples, n_features = len(X), len(X[0])
        
        if self.init_method == 'random':
            # Random initialization
            centroids = []
            for _ in range(self.k):
                centroid = [random.uniform(min(X[i][j] for i in range(n_samples)), 
                                         max(X[i][j] for i in range(n_samples))) 
                           for j in range(n_features)]
                centroids.append(centroid)
            return centroids
            
        elif self.init_method == 'kmeans++':
            # K-means++ initialization
            centroids = []
            
            # Choose first centroid randomly
            first_centroid = X[random.randint(0, n_samples - 1)].copy()
            centroids.append(first_centroid)
            
            # Choose remaining centroids
            for _ in range(1, self.k):
                distances = []
                
                for point in X:
                    # Find distance to nearest existing centroid
                    min_dist = min(self.euclidean_distance(point, centroid) for centroid in centroids)
                    distances.append(min_dist ** 2)
                
                # Choose next centroid with probability proportional to squared distance
                total_distance = sum(distances)
                if total_distance == 0:
                    # If all distances are 0, choose randomly
                    new_centroid = X[random.randint(0, n_samples - 1)].copy()
                else:
                    probabilities = [d / total_distance for d in distances]
                    
                    # Weighted random selection
                    rand_val = random.random()
                    cumulative_prob = 0
                    
                    for i, prob in enumerate(probabilities):
                        cumulative_prob += prob
                        if rand_val <= cumulative_prob:
                            new_centroid = X[i].copy()
                            break
                    else:
                        new_centroid = X[-1].copy()
                
                centroids.append(new_centroid)
            
            return centroids
            
        elif self.init_method == 'forgy':
            # Forgy initialization: choose k random data points
            indices = random.sample(range(n_samples), self.k)
            return [X[i].copy() for i in indices]
            
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def assign_clusters(self, X: List[List[float]], centroids: List[List[float]]) -> List[int]:
        """Assign each point to the nearest centroid"""
        assignments = []
        
        for point in X:
            distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            assignments.append(closest_centroid)
        
        return assignments
    
    def update_centroids(self, X: List[List[float]], assignments: List[int]) -> List[List[float]]:
        """Update centroids based on current cluster assignments"""
        n_features = len(X[0])
        new_centroids = []
        
        for cluster_id in range(self.k):
            # Find all points assigned to this cluster
            cluster_points = [X[i] for i in range(len(X)) if assignments[i] == cluster_id]
            
            if cluster_points:
                # Calculate mean of cluster points
                centroid = [sum(point[j] for point in cluster_points) / len(cluster_points) 
                          for j in range(n_features)]
            else:
                # If no points assigned, reinitialize randomly
                centroid = [random.uniform(min(X[i][j] for i in range(len(X))), 
                                         max(X[i][j] for i in range(len(X)))) 
                          for j in range(n_features)]
            
            new_centroids.append(centroid)
        
        return new_centroids
    
    def calculate_inertia(self, X: List[List[float]], centroids: List[List[float]], assignments: List[int]) -> float:
        """Calculate within-cluster sum of squares (inertia)"""
        total_inertia = 0.0
        
        for i, point in enumerate(X):
            cluster_id = assignments[i]
            centroid = centroids[cluster_id]
            distance = self.euclidean_distance(point, centroid)
            total_inertia += distance ** 2
        
        return total_inertia
    
    def fit(self, X: List[List[float]]):
        """Fit K-means clustering to data"""
        print(f"Starting K-means clustering with k={self.k}...")
        
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Empty dataset provided")
        
        if self.k > n_samples:
            raise ValueError(f"k ({self.k}) cannot be greater than number of samples ({n_samples})")
        
        # Initialize centroids
        print(f"Initializing centroids using {self.init_method} method...")
        self.centroids = self.initialize_centroids(X)
        
        # Main K-means loop
        self.inertia_history = []
        previous_inertia = float('inf')
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            assignments = self.assign_clusters(X, self.centroids)
            
            # Calculate current inertia
            current_inertia = self.calculate_inertia(X, self.centroids, assignments)
            self.inertia_history.append(current_inertia)
            
            # Check for convergence
            inertia_change = abs(previous_inertia - current_inertia)
            if inertia_change < self.tolerance:
                print(f"Converged after {iteration + 1} iterations (inertia change: {inertia_change:.6f})")
                self.converged = True
                self.n_iterations = iteration + 1
                break
            
            # Update centroids
            new_centroids = self.update_centroids(X, assignments)
            
            # Check if centroids changed significantly
            centroid_change = max(self.euclidean_distance(old, new) 
                                for old, new in zip(self.centroids, new_centroids))
            
            if centroid_change < self.tolerance:
                print(f"Converged after {iteration + 1} iterations (centroid change: {centroid_change:.6f})")
                self.converged = True
                self.n_iterations = iteration + 1
                break
            
            self.centroids = new_centroids
            previous_inertia = current_inertia
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Inertia = {current_inertia:.4f}")
        
        if not self.converged:
            print(f"Reached maximum iterations ({self.max_iters}) without convergence")
            self.n_iterations = self.max_iters
        
        # Final assignments
        self.labels = self.assign_clusters(X, self.centroids)
        self.cluster_assignments = self.labels
        
        print(f"Final inertia: {self.inertia_history[-1]:.4f}")
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster assignments for new data"""
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return self.assign_clusters(X, self.centroids)
    
    def get_cluster_info(self, X: List[List[float]]) -> Dict[str, Any]:
        """Get detailed information about clusters"""
        if self.labels is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        cluster_info = {}
        
        for cluster_id in range(self.k):
            cluster_points = [X[i] for i in range(len(X)) if self.labels[i] == cluster_id]
            cluster_size = len(cluster_points)
            
            if cluster_points:
                # Calculate cluster statistics
                n_features = len(cluster_points[0])
                
                # Mean values
                mean_values = [sum(point[j] for point in cluster_points) / cluster_size 
                             for j in range(n_features)]
                
                # Standard deviations
                std_values = []
                for j in range(n_features):
                    variance = sum((point[j] - mean_values[j]) ** 2 for point in cluster_points) / cluster_size
                    std_values.append(math.sqrt(variance))
                
                # Intra-cluster distances
                distances = [self.euclidean_distance(point, self.centroids[cluster_id]) 
                           for point in cluster_points]
                
                cluster_info[cluster_id] = {
                    'size': cluster_size,
                    'centroid': self.centroids[cluster_id].copy(),
                    'mean_values': mean_values,
                    'std_values': std_values,
                    'avg_distance_to_centroid': sum(distances) / len(distances) if distances else 0,
                    'max_distance_to_centroid': max(distances) if distances else 0,
                    'min_distance_to_centroid': min(distances) if distances else 0
                }
            else:
                cluster_info[cluster_id] = {
                    'size': 0,
                    'centroid': self.centroids[cluster_id].copy(),
                    'mean_values': [],
                    'std_values': [],
                    'avg_distance_to_centroid': 0,
                    'max_distance_to_centroid': 0,
                    'min_distance_to_centroid': 0
                }
        
        return cluster_info

class ElbowMethod:
    """Implement Elbow Method for optimal k selection"""
    
    @staticmethod
    def find_optimal_k(X: List[List[float]], k_range: range = None, max_iters: int = 50) -> Tuple[List[int], List[float]]:
        """Find optimal k using elbow method"""
        if k_range is None:
            k_range = range(1, min(11, len(X) + 1))
        
        k_values = []
        inertias = []
        
        print("\nRunning Elbow Method to find optimal k...")
        
        for k in k_range:
            if k >= len(X):
                break
                
            print(f"Testing k={k}...")
            
            kmeans = KMeansClusterer(k=k, max_iters=max_iters, init_method='kmeans++')
            kmeans.fit(X)
            
            k_values.append(k)
            inertias.append(kmeans.inertia_history[-1] if kmeans.inertia_history else 0)
        
        return k_values, inertias
    
    @staticmethod
    def calculate_elbow_score(inertias: List[float]) -> List[float]:
        """Calculate elbow scores (rate of change in inertia reduction)"""
        if len(inertias) < 3:
            return [0] * len(inertias)
        
        # Calculate second derivative (curvature)
        elbow_scores = [0, 0]  # First two values are undefined
        
        for i in range(2, len(inertias)):
            # Second derivative approximation
            second_derivative = inertias[i-2] - 2*inertias[i-1] + inertias[i]
            elbow_scores.append(abs(second_derivative))
        
        return elbow_scores

class CustomerDataGenerator:
    """Generate realistic customer segmentation dataset"""
    
    @staticmethod
    def generate_customer_data(n_customers: int = 500) -> Tuple[List[List[float]], List[str], List[str]]:
        """Generate synthetic customer behavior dataset"""
        random.seed(42)  # For reproducible results
        
        feature_names = [
            'annual_spending', 'visit_frequency', 'avg_transaction_value', 
            'customer_age', 'loyalty_years', 'online_purchases_ratio',
            'seasonal_spending_variance', 'return_rate', 'referral_count'
        ]
        
        customers_data = []
        customer_profiles = []  # For analysis purposes
        
        # Define customer segments with different characteristics
        segments = {
            'high_value': {
                'weight': 0.15,  # 15% of customers
                'spending': (8000, 15000),
                'frequency': (15, 25),
                'transaction': (200, 500),
                'age': (35, 55),
                'loyalty': (3, 8),
                'online_ratio': (0.6, 0.9),
                'variance': (0.2, 0.4),
                'return_rate': (0.02, 0.08),
                'referrals': (3, 8)
            },
            'regular': {
                'weight': 0.45,  # 45% of customers
                'spending': (2000, 6000),
                'frequency': (8, 15),
                'transaction': (80, 200),
                'age': (25, 45),
                'loyalty': (1, 4),
                'online_ratio': (0.3, 0.7),
                'variance': (0.15, 0.35),
                'return_rate': (0.05, 0.12),
                'referrals': (1, 4)
            },
            'occasional': {
                'weight': 0.25,  # 25% of customers
                'spending': (500, 2500),
                'frequency': (3, 8),
                'transaction': (40, 100),
                'age': (20, 35),
                'loyalty': (0.5, 2),
                'online_ratio': (0.1, 0.5),
                'variance': (0.25, 0.5),
                'return_rate': (0.08, 0.18),
                'referrals': (0, 2)
            },
            'bargain_hunters': {
                'weight': 0.15,  # 15% of customers
                'spending': (300, 1500),
                'frequency': (5, 12),
                'transaction': (20, 60),
                'age': (25, 50),
                'loyalty': (0.5, 3),
                'online_ratio': (0.4, 0.8),
                'variance': (0.4, 0.7),
                'return_rate': (0.12, 0.25),
                'referrals': (0, 1)
            }
        }
        
        for _ in range(n_customers):
            # Select customer segment based on weights
            rand_val = random.random()
            cumulative_weight = 0
            selected_segment = 'regular'
            
            for segment_name, segment_data in segments.items():
                cumulative_weight += segment_data['weight']
                if rand_val <= cumulative_weight:
                    selected_segment = segment_name
                    break
            
            segment = segments[selected_segment]
            
            # Generate customer features based on segment
            annual_spending = random.uniform(*segment['spending'])
            visit_frequency = random.uniform(*segment['frequency'])
            avg_transaction = annual_spending / visit_frequency
            customer_age = random.uniform(*segment['age'])
            loyalty_years = random.uniform(*segment['loyalty'])
            online_ratio = random.uniform(*segment['online_ratio'])
            seasonal_variance = random.uniform(*segment['variance'])
            return_rate = random.uniform(*segment['return_rate'])
            referral_count = random.randint(int(segment['referrals'][0]), int(segment['referrals'][1]))
            
            # Add some noise and correlations
            if customer_age > 40:
                online_ratio *= random.uniform(0.7, 0.9)  # Older customers less online
            
            if loyalty_years > 3:
                return_rate *= random.uniform(0.5, 0.8)  # Loyal customers return less
                referral_count += random.randint(0, 2)  # More referrals
            
            # Ensure realistic transaction value
            avg_transaction = max(10, min(avg_transaction, annual_spending / 2))
            
            customer_features = [
                annual_spending,
                visit_frequency,
                avg_transaction,
                customer_age,
                loyalty_years,
                online_ratio,
                seasonal_variance,
                return_rate,
                float(referral_count)
            ]
            
            customers_data.append(customer_features)
            customer_profiles.append(selected_segment)
        
        return customers_data, feature_names, customer_profiles

class ClusterEvaluator:
    """Comprehensive evaluation metrics for clustering"""
    
    @staticmethod
    def silhouette_score(X: List[List[float]], labels: List[int]) -> float:
        """Calculate average silhouette score"""
        n_samples = len(X)
        if n_samples <= 1:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n_samples):
            # Calculate average distance to points in same cluster (a)
            same_cluster_distances = []
            for j in range(n_samples):
                if i != j and labels[i] == labels[j]:
                    distance = math.sqrt(sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i]))))
                    same_cluster_distances.append(distance)
            
            if same_cluster_distances:
                a = sum(same_cluster_distances) / len(same_cluster_distances)
            else:
                a = 0
            
            # Calculate minimum average distance to points in other clusters (b)
            other_cluster_distances = {}
            for j in range(n_samples):
                if i != j and labels[i] != labels[j]:
                    cluster_id = labels[j]
                    distance = math.sqrt(sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i]))))
                    
                    if cluster_id not in other_cluster_distances:
                        other_cluster_distances[cluster_id] = []
                    other_cluster_distances[cluster_id].append(distance)
            
            if other_cluster_distances:
                avg_distances = [sum(distances) / len(distances) 
                               for distances in other_cluster_distances.values()]
                b = min(avg_distances)
            else:
                b = 0
            
            # Calculate silhouette score for this point
            if max(a, b) == 0:
                silhouette_scores.append(0)
            else:
                silhouette_scores.append((b - a) / max(a, b))
        
        return sum(silhouette_scores) / len(silhouette_scores)
    
    @staticmethod
    def davies_bouldin_score(X: List[List[float]], labels: List[int], centroids: List[List[float]]) -> float:
        """Calculate Davies-Bouldin score (lower is better)"""
        n_clusters = len(centroids)
        if n_clusters <= 1:
            return 0.0
        
        # Calculate average intra-cluster distances
        cluster_dispersions = []
        
        for k in range(n_clusters):
            cluster_points = [X[i] for i in range(len(X)) if labels[i] == k]
            
            if cluster_points:
                avg_distance = sum(math.sqrt(sum((point[j] - centroids[k][j]) ** 2 
                                               for j in range(len(point)))) 
                                 for point in cluster_points) / len(cluster_points)
                cluster_dispersions.append(avg_distance)
            else:
                cluster_dispersions.append(0)
        
        # Calculate Davies-Bouldin score
        db_scores = []
        
        for i in range(n_clusters):
            max_ratio = 0
            
            for j in range(n_clusters):
                if i != j:
                    # Distance between centroids
                    centroid_distance = math.sqrt(sum((centroids[i][k] - centroids[j][k]) ** 2 
                                                     for k in range(len(centroids[i]))))
                    
                    if centroid_distance > 0:
                        ratio = (cluster_dispersions[i] + cluster_dispersions[j]) / centroid_distance
                        max_ratio = max(max_ratio, ratio)
            
            db_scores.append(max_ratio)
        
        return sum(db_scores) / len(db_scores)
    
    @staticmethod
    def calinski_harabasz_score(X: List[List[float]], labels: List[int]) -> float:
        """Calculate Calinski-Harabasz score (higher is better)"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        n_clusters = len(set(labels))
        
        if n_clusters <= 1 or n_samples <= n_clusters:
            return 0.0
        
        # Calculate overall mean
        overall_mean = [sum(X[i][j] for i in range(n_samples)) / n_samples 
                       for j in range(n_features)]
        
        # Calculate between-cluster sum of squares
        between_ss = 0
        for cluster_id in set(labels):
            cluster_points = [X[i] for i in range(n_samples) if labels[i] == cluster_id]
            cluster_size = len(cluster_points)
            
            if cluster_size > 0:
                cluster_mean = [sum(point[j] for point in cluster_points) / cluster_size 
                              for j in range(n_features)]
                
                cluster_between = sum((cluster_mean[j] - overall_mean[j]) ** 2 
                                    for j in range(n_features))
                between_ss += cluster_size * cluster_between
        
        # Calculate within-cluster sum of squares
        within_ss = 0
        for i, point in enumerate(X):
            cluster_id = labels[i]
            cluster_points = [X[j] for j in range(n_samples) if labels[j] == cluster_id]
            cluster_mean = [sum(cp[k] for cp in cluster_points) / len(cluster_points) 
                          for k in range(n_features)]
            
            point_within = sum((point[j] - cluster_mean[j]) ** 2 for j in range(n_features))
            within_ss += point_within
        
        if within_ss == 0:
            return 0.0
        
        ch_score = (between_ss / (n_clusters - 1)) / (within_ss / (n_samples - n_clusters))
        return ch_score

def split_data(X: List[List[float]], train_ratio: float = 0.8) -> Tuple[List[List[float]], List[List[float]]]:
    """Split data for clustering (unsupervised, so no labels)"""
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    
    # Shuffle indices
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    
    return X_train, X_test

def print_data_analysis(X: List[List[float]], feature_names: List[str], true_profiles: List[str] = None):
    """Print comprehensive data analysis"""
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION DATASET ANALYSIS")
    print("=" * 60)
    
    n_samples = len(X)
    n_features = len(X[0]) if X else 0
    
    print(f"Number of customers: {n_samples}")
    print(f"Number of features: {n_features}")
    print(f"Customer features: {', '.join(feature_names[:5])}...")
    
    # True profile distribution (if available)
    if true_profiles:
        from collections import Counter
        profile_counts = Counter(true_profiles)
        print(f"\nTrue Customer Profiles:")
        for profile, count in sorted(profile_counts.items()):
            percentage = count / n_samples * 100
            print(f"{profile.replace('_', ' ').title():15s}: {count:3d} customers ({percentage:5.1f}%)")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"{'Feature':<25} {'Mean':<10} {'Min':<10} {'Max':<10} {'Std':<10}")
    print("-" * 70)
    
    for j in range(n_features):
        feature_values = [X[i][j] for i in range(n_samples)]
        mean_val = sum(feature_values) / len(feature_values)
        min_val = min(feature_values)
        max_val = max(feature_values)
        
        # Calculate standard deviation
        variance = sum((x - mean_val) ** 2 for x in feature_values) / len(feature_values)
        std_val = math.sqrt(variance)
        
        print(f"{feature_names[j]:<25} {mean_val:<10.2f} {min_val:<10.2f} {max_val:<10.2f} {std_val:<10.2f}")

def print_clustering_results(kmeans: KMeansClusterer, X: List[List[float]], feature_names: List[str]):
    """Print comprehensive clustering results"""
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING RESULTS")
    print("=" * 60)
    
    # Model information
    print(f"\nModel Configuration:")
    print(f"Number of clusters (k): {kmeans.k}")
    print(f"Initialization method: {kmeans.init_method}")
    print(f"Converged: {kmeans.converged}")
    print(f"Number of iterations: {kmeans.n_iterations}")
    print(f"Final inertia: {kmeans.inertia_history[-1]:.4f}")
    
    # Cluster information
    cluster_info = kmeans.get_cluster_info(X)
    
    print(f"\nCluster Analysis:")
    for cluster_id in range(kmeans.k):
        info = cluster_info[cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {info['size']} customers ({info['size']/len(X)*100:.1f}%)")
        print(f"  Avg distance to centroid: {info['avg_distance_to_centroid']:.3f}")
        
        # Show top characteristics
        print(f"  Key characteristics:")
        centroid = info['centroid']
        for j in range(min(5, len(centroid))):  # Show first 5 features
            print(f"    {feature_names[j]:<25}: {centroid[j]:.2f}")
    
    # Evaluation metrics
    evaluator = ClusterEvaluator()
    silhouette = evaluator.silhouette_score(X, kmeans.labels)
    davies_bouldin = evaluator.davies_bouldin_score(X, kmeans.labels, kmeans.centroids)
    calinski_harabasz = evaluator.calinski_harabasz_score(X, kmeans.labels)
    
    print(f"\nClustering Quality Metrics:")
    print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f} (higher is better)")

def print_elbow_analysis(k_values: List[int], inertias: List[float]):
    """Print elbow method analysis"""
    print(f"\nElbow Method Analysis:")
    print(f"{'k':<4} {'Inertia':<12} {'Reduction':<12} {'% Change':<10}")
    print("-" * 40)
    
    for i, (k, inertia) in enumerate(zip(k_values, inertias)):
        if i == 0:
            reduction = 0
            pct_change = 0
        else:
            reduction = inertias[i-1] - inertia
            pct_change = (reduction / inertias[i-1] * 100) if inertias[i-1] > 0 else 0
        
        print(f"{k:<4} {inertia:<12.2f} {reduction:<12.2f} {pct_change:<10.1f}%")
    
    # Calculate elbow scores
    elbow_scores = ElbowMethod.calculate_elbow_score(inertias)
    if len(elbow_scores) > 2:
        optimal_k_idx = elbow_scores.index(max(elbow_scores[2:]))
        optimal_k = k_values[optimal_k_idx]
        print(f"\nSuggested optimal k based on elbow method: {optimal_k}")

def main():
    """Main function to demonstrate K-means clustering for customer segmentation"""
    print_header()
    
    # Generate customer dataset
    print("Generating customer segmentation dataset...")
    X, feature_names, true_profiles = CustomerDataGenerator.generate_customer_data(n_customers=400)
    
    # Analyze dataset
    print_data_analysis(X, feature_names, true_profiles)
    
    # Find optimal k using elbow method
    print("\n" + "=" * 60)
    print("ELBOW METHOD FOR OPTIMAL K SELECTION")
    print("=" * 60)
    
    k_values, inertias = ElbowMethod.find_optimal_k(X, range(1, 8), max_iters=30)
    print_elbow_analysis(k_values, inertias)
    
    # Apply K-means with optimal k
    optimal_k = 4  # Based on known true segments
    print(f"\n" + "=" * 60)
    print(f"APPLYING K-MEANS WITH K={optimal_k}")
    print("=" * 60)
    
    kmeans = KMeansClusterer(
        k=optimal_k,
        max_iters=100,
        tolerance=1e-4,
        init_method='kmeans++'
    )
    
    kmeans.fit(X)
    
    # Analyze results
    print_clustering_results(kmeans, X, feature_names)
    
    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENTATION PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'model': kmeans,
        'data': X,
        'feature_names': feature_names,
        'true_profiles': true_profiles,
        'elbow_analysis': (k_values, inertias),
        'metrics': {
            'silhouette': ClusterEvaluator.silhouette_score(X, kmeans.labels),
            'inertia': kmeans.inertia_history[-1]
        }
    }

if __name__ == "__main__":
    results = main()

# Dependencies and Notes:
# This project implements K-Means clustering completely from scratch using only Python built-ins.
# 
# Key Dependencies:
# - math: For mathematical operations like sqrt(), exp()
# - random: For centroid initialization, data generation, and shuffling
# - typing: For type hints (List, Tuple, Dict, Any, Optional)
# 
# Educational Notes:
# 1. K-means partitions data into k clusters by minimizing within-cluster sum of squares
# 2. Algorithm alternates between assigning points to nearest centroids and updating centroids
# 3. K-means++ initialization improves convergence and final clustering quality
# 4. Elbow method helps determine optimal number of clusters
# 5. Silhouette analysis measures how similar objects are to their own cluster vs other clusters
# 6. Multiple distance metrics (Euclidean, Manhattan, Cosine) can be used
#
# This implementation demonstrates:
# - Complete K-means algorithm with multiple initialization methods
# - Customer segmentation as a practical business application
# - Elbow method for optimal k selection with statistical analysis
# - Comprehensive cluster evaluation using multiple quality metrics
# - Realistic customer data generation with behavioral patterns
# - Business insights through cluster analysis and interpretation
# - Unsupervised learning principles and their real-world applications
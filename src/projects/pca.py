# PCA: Dimensionality Reduction
# Principal Component Analysis for data visualization

import math
import random
from typing import List, Tuple

class PCA:
    """
    Principal Component Analysis implementation.
    Reduces high-dimensional data to lower dimensions while preserving variance.
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def standardize(self, X: List[List[float]]) -> List[List[float]]:
        """
        Standardize the dataset.
        
        Args:
            X: Input data matrix
            
        Returns:
            Standardized data
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        # Calculate mean for each feature
        self.mean = [0.0] * n_features
        for i in range(n_samples):
            for j in range(n_features):
                self.mean[j] += X[i][j]
        self.mean = [m / n_samples for m in self.mean]
        
        # Center the data
        X_centered = []
        for i in range(n_samples):
            row = [X[i][j] - self.mean[j] for j in range(n_features)]
            X_centered.append(row)
        
        return X_centered
    
    def compute_covariance_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """
        Compute covariance matrix.
        
        Args:
            X: Centered data matrix
            
        Returns:
            Covariance matrix
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        cov_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
        
        for i in range(n_features):
            for j in range(n_features):
                cov = 0.0
                for k in range(n_samples):
                    cov += X[k][i] * X[k][j]
                cov_matrix[i][j] = cov / (n_samples - 1)
        
        return cov_matrix
    
    def power_iteration(self, matrix: List[List[float]], num_iterations: int = 100) -> Tuple[List[float], float]:
        """
        Find dominant eigenvector using power iteration.
        
        Args:
            matrix: Square matrix
            num_iterations: Number of iterations
            
        Returns:
            Tuple of (eigenvector, eigenvalue)
        """
        n = len(matrix)
        # Random initialization
        v = [random.random() for _ in range(n)]
        
        for _ in range(num_iterations):
            # Matrix-vector multiplication
            Av = [0.0] * n
            for i in range(n):
                for j in range(n):
                    Av[i] += matrix[i][j] * v[j]
            
            # Normalize
            norm = math.sqrt(sum(x**2 for x in Av))
            v = [x / norm for x in Av]
        
        # Calculate eigenvalue
        Av = [0.0] * n
        for i in range(n):
            for j in range(n):
                Av[i] += matrix[i][j] * v[j]
        
        eigenvalue = sum(Av[i] * v[i] for i in range(n))
        
        return v, eigenvalue
    
    def deflate_matrix(self, matrix: List[List[float]], eigenvector: List[float], eigenvalue: float) -> List[List[float]]:
        """
        Remove the dominant eigenvector from matrix (deflation).
        
        Args:
            matrix: Current matrix
            eigenvector: Eigenvector to remove
            eigenvalue: Corresponding eigenvalue
            
        Returns:
            Deflated matrix
        """
        n = len(matrix)
        deflated = [[matrix[i][j] for j in range(n)] for i in range(n)]
        
        for i in range(n):
            for j in range(n):
                deflated[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
        
        return deflated
    
    def fit(self, X: List[List[float]]):
        """
        Fit PCA model to data.
        
        Args:
            X: Input data matrix (samples x features)
        """
        # Standardize data
        X_centered = self.standardize(X)
        
        # Compute covariance matrix
        cov_matrix = self.compute_covariance_matrix(X_centered)
        
        # Find principal components using power iteration
        self.components = []
        self.explained_variance = []
        
        current_matrix = [row[:] for row in cov_matrix]
        
        for _ in range(self.n_components):
            eigenvector, eigenvalue = self.power_iteration(current_matrix)
            self.components.append(eigenvector)
            self.explained_variance.append(eigenvalue)
            
            # Deflate matrix to find next component
            current_matrix = self.deflate_matrix(current_matrix, eigenvector, eigenvalue)
        
        # Calculate explained variance ratio
        total_variance = sum(self.explained_variance)
        self.explained_variance_ratio = [ev / total_variance for ev in self.explained_variance]
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Transform data to principal component space.
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data (samples x n_components)
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        # Center the data using training mean
        X_centered = []
        for i in range(n_samples):
            row = [X[i][j] - self.mean[j] for j in range(n_features)]
            X_centered.append(row)
        
        # Project onto principal components
        X_transformed = []
        for i in range(n_samples):
            sample_transformed = []
            for component in self.components:
                # Dot product
                projection = sum(X_centered[i][j] * component[j] for j in range(n_features))
                sample_transformed.append(projection)
            X_transformed.append(sample_transformed)
        
        return X_transformed
    
    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)


def generate_high_dimensional_data(n_samples: int = 500, n_features: int = 50) -> Tuple:
    """
    Generate synthetic high-dimensional dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Tuple of (data, labels)
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate 3 clusters
        cluster = random.randint(0, 2)
        
        # Base point for cluster
        if cluster == 0:
            base = [1.0, 2.0]
        elif cluster == 1:
            base = [-1.5, -1.0]
        else:
            base = [0.5, -2.0]
        
        # Generate features
        sample = []
        for i in range(n_features):
            if i < 2:
                # First two features are informative
                value = base[i] + random.gauss(0, 0.5)
            else:
                # Rest are noise
                value = random.gauss(0, 1.0)
            sample.append(value)
        
        X.append(sample)
        y.append(cluster)
    
    return X, y


if __name__ == "__main__":
    print("PCA: Dimensionality Reduction")
    print("="*60)
    
    # Generate high-dimensional data
    print("\nGenerating high-dimensional dataset...")
    X, y = generate_high_dimensional_data(n_samples=500, n_features=50)
    print(f"Dataset shape: {len(X)} samples x {len(X[0])} features")
    
    # Apply PCA
    print("\nApplying PCA to reduce to 2 dimensions...")
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    print(f"\nExplained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio):
        print(f"  PC{i+1}: {ratio*100:.2f}%")
    
    total_explained = sum(pca.explained_variance_ratio) * 100
    print(f"\nTotal variance explained: {total_explained:.2f}%")
    
    print("\nPrincipal Components (first 5 coefficients):")
    for i, component in enumerate(pca.components):
        print(f"  PC{i+1}: {component[:5]}")
    
    print("\n" + "="*60)
    print("PCA Complete! Data reduced from 50D to 2D")

"""
Hierarchical Clustering: Gene Expression Analysis
Comprehensive implementation of hierarchical clustering for analyzing gene expression patterns
CBSE Class 12 AI Project
"""

import json
import math
from typing import List, Tuple, Dict, Any

class HierarchicalClustering:
    """
    Hierarchical Clustering implementation with multiple linkage methods
    """
    
    def __init__(self, linkage='ward'):
        """
        Initialize hierarchical clustering
        
        Args:
            linkage: Linkage method ('single', 'complete', 'average', 'ward')
        """
        self.linkage = linkage
        self.dendrogram = []
        self.clusters = []
        self.distance_matrix = []
        
    def euclidean_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def calculate_distance_matrix(self, data: List[List[float]]) -> List[List[float]]:
        """Calculate pairwise distance matrix"""
        n = len(data)
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.euclidean_distance(data[i], data[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        
        return matrix
    
    def find_closest_clusters(self, distance_matrix: List[List[float]], 
                             active_clusters: List[int]) -> Tuple[int, int, float]:
        """Find the two closest clusters"""
        min_dist = float('inf')
        cluster1, cluster2 = -1, -1
        
        for i in range(len(active_clusters)):
            for j in range(i + 1, len(active_clusters)):
                idx1 = active_clusters[i]
                idx2 = active_clusters[j]
                
                if distance_matrix[idx1][idx2] < min_dist:
                    min_dist = distance_matrix[idx1][idx2]
                    cluster1 = i
                    cluster2 = j
        
        return cluster1, cluster2, min_dist
    
    def single_linkage(self, cluster1_indices: List[int], cluster2_indices: List[int],
                      distance_matrix: List[List[float]]) -> float:
        """Single linkage: minimum distance"""
        min_dist = float('inf')
        for i in cluster1_indices:
            for j in cluster2_indices:
                min_dist = min(min_dist, distance_matrix[i][j])
        return min_dist
    
    def complete_linkage(self, cluster1_indices: List[int], cluster2_indices: List[int],
                        distance_matrix: List[List[float]]) -> float:
        """Complete linkage: maximum distance"""
        max_dist = 0.0
        for i in cluster1_indices:
            for j in cluster2_indices:
                max_dist = max(max_dist, distance_matrix[i][j])
        return max_dist
    
    def average_linkage(self, cluster1_indices: List[int], cluster2_indices: List[int],
                       distance_matrix: List[List[float]]) -> float:
        """Average linkage: mean distance"""
        total_dist = 0.0
        count = 0
        for i in cluster1_indices:
            for j in cluster2_indices:
                total_dist += distance_matrix[i][j]
                count += 1
        return total_dist / count if count > 0 else 0.0
    
    def ward_linkage(self, cluster1_indices: List[int], cluster2_indices: List[int],
                    data: List[List[float]]) -> float:
        """Ward linkage: minimize within-cluster variance"""
        # Calculate centroids
        centroid1 = self.calculate_centroid([data[i] for i in cluster1_indices])
        centroid2 = self.calculate_centroid([data[i] for i in cluster2_indices])
        merged_centroid = self.calculate_centroid([data[i] for i in cluster1_indices + cluster2_indices])
        
        # Calculate increase in variance
        n1 = len(cluster1_indices)
        n2 = len(cluster2_indices)
        
        sse_increase = (n1 * n2) / (n1 + n2) * self.euclidean_distance(centroid1, centroid2) ** 2
        return math.sqrt(sse_increase)
    
    def calculate_centroid(self, points: List[List[float]]) -> List[float]:
        """Calculate centroid of points"""
        if not points:
            return []
        
        n_features = len(points[0])
        centroid = [0.0] * n_features
        
        for point in points:
            for i in range(n_features):
                centroid[i] += point[i]
        
        return [c / len(points) for c in centroid]
    
    def update_distance_matrix(self, distance_matrix: List[List[float]], 
                              cluster_members: List[List[int]],
                              merged_cluster: int, removed_cluster: int,
                              data: List[List[float]]) -> List[List[float]]:
        """Update distance matrix after merging clusters"""
        n = len(distance_matrix)
        
        # Update distances for the merged cluster
        for i in range(len(cluster_members)):
            if i == merged_cluster or i == removed_cluster:
                continue
            
            if self.linkage == 'single':
                new_dist = self.single_linkage(
                    cluster_members[merged_cluster],
                    cluster_members[i],
                    distance_matrix
                )
            elif self.linkage == 'complete':
                new_dist = self.complete_linkage(
                    cluster_members[merged_cluster],
                    cluster_members[i],
                    distance_matrix
                )
            elif self.linkage == 'average':
                new_dist = self.average_linkage(
                    cluster_members[merged_cluster],
                    cluster_members[i],
                    distance_matrix
                )
            elif self.linkage == 'ward':
                new_dist = self.ward_linkage(
                    cluster_members[merged_cluster],
                    cluster_members[i],
                    data
                )
            else:
                new_dist = self.average_linkage(
                    cluster_members[merged_cluster],
                    cluster_members[i],
                    distance_matrix
                )
            
            # Update both symmetric positions
            for idx1 in cluster_members[merged_cluster]:
                for idx2 in cluster_members[i]:
                    distance_matrix[idx1][idx2] = new_dist
                    distance_matrix[idx2][idx1] = new_dist
        
        return distance_matrix
    
    def fit(self, data: List[List[float]], labels: List[str] = None) -> Dict[str, Any]:
        """
        Perform hierarchical clustering
        
        Args:
            data: List of data points
            labels: Optional labels for data points
        
        Returns:
            Dictionary containing dendrogram and cluster information
        """
        n_samples = len(data)
        
        if labels is None:
            labels = [f"Sample_{i}" for i in range(n_samples)]
        
        # Initialize distance matrix
        self.distance_matrix = self.calculate_distance_matrix(data)
        
        # Initialize cluster members
        cluster_members = [[i] for i in range(n_samples)]
        active_clusters = list(range(n_samples))
        
        # Store dendrogram information
        self.dendrogram = []
        merge_history = []
        
        step = 0
        while len(active_clusters) > 1:
            # Find closest clusters
            idx1, idx2, distance = self.find_closest_clusters(
                self.distance_matrix, 
                active_clusters
            )
            
            cluster1 = active_clusters[idx1]
            cluster2 = active_clusters[idx2]
            
            # Record merge
            merge_info = {
                'step': step,
                'cluster1': cluster1,
                'cluster2': cluster2,
                'distance': distance,
                'members1': cluster_members[cluster1].copy(),
                'members2': cluster_members[cluster2].copy(),
                'labels1': [labels[i] for i in cluster_members[cluster1]],
                'labels2': [labels[i] for i in cluster_members[cluster2]]
            }
            self.dendrogram.append(merge_info)
            
            # Merge clusters
            cluster_members[cluster1].extend(cluster_members[cluster2])
            
            # Update distance matrix
            self.distance_matrix = self.update_distance_matrix(
                self.distance_matrix,
                cluster_members,
                cluster1,
                cluster2,
                data
            )
            
            # Remove merged cluster
            active_clusters.remove(cluster2)
            
            step += 1
        
        # Final cluster assignment
        self.clusters = cluster_members[active_clusters[0]]
        
        return {
            'dendrogram': self.dendrogram,
            'final_cluster': self.clusters,
            'n_merges': len(self.dendrogram)
        }
    
    def cut_tree(self, n_clusters: int) -> List[int]:
        """
        Cut dendrogram to get specified number of clusters
        
        Args:
            n_clusters: Desired number of clusters
        
        Returns:
            Cluster assignments for each sample
        """
        if n_clusters <= 0 or n_clusters > len(self.clusters):
            return list(range(len(self.clusters)))
        
        # Work backwards through dendrogram
        n_merges = len(self.dendrogram)
        cuts = n_merges - n_clusters + 1
        
        # Initialize cluster assignments
        assignments = list(range(len(self.clusters)))
        cluster_id = 0
        
        # Apply cuts
        for i in range(cuts):
            merge = self.dendrogram[i]
            for member in merge['members1'] + merge['members2']:
                assignments[member] = cluster_id
            cluster_id += 1
        
        return assignments


def generate_gene_expression_data():
    """Generate synthetic gene expression data"""
    import random
    random.seed(42)
    
    # Generate gene expression profiles
    genes = []
    labels = []
    
    # Group 1: High expression in conditions 1-2
    for i in range(8):
        gene = [
            random.uniform(8, 10),  # Condition 1
            random.uniform(7, 9),   # Condition 2
            random.uniform(1, 3),   # Condition 3
            random.uniform(1, 2),   # Condition 4
        ]
        genes.append(gene)
        labels.append(f"Gene_A{i+1}")
    
    # Group 2: High expression in conditions 3-4
    for i in range(8):
        gene = [
            random.uniform(1, 2),   # Condition 1
            random.uniform(1, 3),   # Condition 2
            random.uniform(8, 10),  # Condition 3
            random.uniform(7, 9),   # Condition 4
        ]
        genes.append(gene)
        labels.append(f"Gene_B{i+1}")
    
    # Group 3: Moderate expression
    for i in range(6):
        gene = [
            random.uniform(4, 6),   # Condition 1
            random.uniform(4, 6),   # Condition 2
            random.uniform(4, 6),   # Condition 3
            random.uniform(4, 6),   # Condition 4
        ]
        genes.append(gene)
        labels.append(f"Gene_C{i+1}")
    
    return genes, labels


def main():
    """Main execution function"""
    print("=" * 70)
    print("Hierarchical Clustering: Gene Expression Analysis")
    print("=" * 70)
    print()
    
    # Generate data
    print("Step 1: Generating Gene Expression Data")
    print("-" * 70)
    genes, labels = generate_gene_expression_data()
    print(f"Generated expression data for {len(genes)} genes")
    print(f"Each gene measured across 4 experimental conditions")
    print()
    
    # Display sample data
    print("Sample Gene Expression Profiles:")
    for i in range(min(5, len(genes))):
        print(f"{labels[i]}: {[f'{x:.2f}' for x in genes[i]]}")
    print()
    
    # Perform clustering with different linkage methods
    linkage_methods = ['single', 'complete', 'average', 'ward']
    
    for method in linkage_methods:
        print(f"\nStep 2: Hierarchical Clustering with {method.upper()} Linkage")
        print("-" * 70)
        
        hc = HierarchicalClustering(linkage=method)
        result = hc.fit(genes, labels)
        
        print(f"Total merges performed: {result['n_merges']}")
        print(f"Linkage method: {method}")
        print()
        
        # Show first few merges
        print("First 5 Merge Steps:")
        for i, merge in enumerate(result['dendrogram'][:5]):
            print(f"  Step {merge['step']}: Merge {merge['labels1']} with {merge['labels2']}")
            print(f"    Distance: {merge['distance']:.4f}")
        print()
        
        # Cut tree to get 3 clusters
        if method == 'ward':
            print("Step 3: Cutting Dendrogram (3 clusters)")
            print("-" * 70)
            assignments = hc.cut_tree(n_clusters=3)
            
            # Group genes by cluster
            clusters = {}
            for idx, cluster_id in enumerate(assignments):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(labels[idx])
            
            for cluster_id, members in sorted(clusters.items()):
                print(f"Cluster {cluster_id + 1}: {len(members)} genes")
                print(f"  Members: {', '.join(members[:5])}")
                if len(members) > 5:
                    print(f"  ... and {len(members) - 5} more")
            print()
    
    # Final summary
    print("\n" + "=" * 70)
    print("Analysis Summary")
    print("=" * 70)
    print(f"✓ Analyzed {len(genes)} gene expression profiles")
    print(f"✓ Tested {len(linkage_methods)} linkage methods")
    print(f"✓ Generated complete dendrogram")
    print(f"✓ Identified gene clusters with similar expression patterns")
    print()
    print("Biological Interpretation:")
    print("- Genes in same cluster show similar expression patterns")
    print("- Dendrogram height indicates dissimilarity between clusters")
    print("- Ward linkage minimizes within-cluster variance")
    print("- Results can guide further functional analysis")
    print()

if __name__ == "__main__":
    main()

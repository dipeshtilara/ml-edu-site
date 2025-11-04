"""
K-Nearest Neighbors: Movie Recommendation System
Comprehensive implementation of KNN for collaborative filtering
CBSE Class 12 AI Project
"""

import json
import math
from typing import List, Tuple, Dict, Any

class KNNRecommender:
    """
    K-Nearest Neighbors Recommender System
    Implements collaborative filtering for movie recommendations
    """
    
    def __init__(self, k: int = 5, similarity_metric: str = 'cosine'):
        """
        Initialize KNN recommender
        
        Args:
            k: Number of nearest neighbors
            similarity_metric: 'cosine', 'euclidean', or 'pearson'
        """
        self.k = k
        self.similarity_metric = similarity_metric
        self.user_ratings = {}
        self.movie_ratings = {}
        self.similarity_cache = {}
        
    def add_rating(self, user_id: int, movie_id: int, rating: float):
        """Add a rating to the system"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        self.user_ratings[user_id][movie_id] = rating
        
        if movie_id not in self.movie_ratings:
            self.movie_ratings[movie_id] = {}
        self.movie_ratings[movie_id][user_id] = rating
    
    def cosine_similarity(self, user1: int, user2: int) -> float:
        """Calculate cosine similarity between two users"""
        # Find common movies
        common_movies = set(self.user_ratings[user1].keys()) & set(self.user_ratings[user2].keys())
        
        if not common_movies:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(
            self.user_ratings[user1][movie] * self.user_ratings[user2][movie]
            for movie in common_movies
        )
        
        magnitude1 = math.sqrt(sum(
            self.user_ratings[user1][movie] ** 2
            for movie in common_movies
        ))
        
        magnitude2 = math.sqrt(sum(
            self.user_ratings[user2][movie] ** 2
            for movie in common_movies
        ))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def euclidean_similarity(self, user1: int, user2: int) -> float:
        """Calculate Euclidean similarity between two users"""
        common_movies = set(self.user_ratings[user1].keys()) & set(self.user_ratings[user2].keys())
        
        if not common_movies:
            return 0.0
        
        # Calculate Euclidean distance
        distance = math.sqrt(sum(
            (self.user_ratings[user1][movie] - self.user_ratings[user2][movie]) ** 2
            for movie in common_movies
        ))
        
        # Convert distance to similarity (0 to 1)
        return 1 / (1 + distance)
    
    def pearson_correlation(self, user1: int, user2: int) -> float:
        """Calculate Pearson correlation between two users"""
        common_movies = set(self.user_ratings[user1].keys()) & set(self.user_ratings[user2].keys())
        
        if len(common_movies) < 2:
            return 0.0
        
        # Calculate means
        mean1 = sum(self.user_ratings[user1][movie] for movie in common_movies) / len(common_movies)
        mean2 = sum(self.user_ratings[user2][movie] for movie in common_movies) / len(common_movies)
        
        # Calculate correlation
        numerator = sum(
            (self.user_ratings[user1][movie] - mean1) * (self.user_ratings[user2][movie] - mean2)
            for movie in common_movies
        )
        
        denominator1 = math.sqrt(sum(
            (self.user_ratings[user1][movie] - mean1) ** 2
            for movie in common_movies
        ))
        
        denominator2 = math.sqrt(sum(
            (self.user_ratings[user2][movie] - mean2) ** 2
            for movie in common_movies
        ))
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)
    
    def calculate_similarity(self, user1: int, user2: int) -> float:
        """Calculate similarity based on chosen metric"""
        # Check cache
        cache_key = (min(user1, user2), max(user1, user2))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Calculate similarity
        if self.similarity_metric == 'cosine':
            similarity = self.cosine_similarity(user1, user2)
        elif self.similarity_metric == 'euclidean':
            similarity = self.euclidean_similarity(user1, user2)
        elif self.similarity_metric == 'pearson':
            similarity = self.pearson_correlation(user1, user2)
        else:
            similarity = self.cosine_similarity(user1, user2)
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def find_k_nearest_neighbors(self, user_id: int) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a user"""
        if user_id not in self.user_ratings:
            return []
        
        similarities = []
        for other_user in self.user_ratings:
            if other_user != user_id:
                similarity = self.calculate_similarity(user_id, other_user)
                similarities.append((other_user, similarity))
        
        # Sort by similarity (descending) and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.k]
    
    def predict_rating(self, user_id: int, movie_id: int) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Predict rating for a movie using KNN
        
        Returns:
            Tuple of (predicted_rating, list_of_neighbors_used)
        """
        if user_id not in self.user_ratings:
            return 0.0, []
        
        # Find k nearest neighbors
        neighbors = self.find_k_nearest_neighbors(user_id)
        
        # Filter neighbors who have rated this movie
        relevant_neighbors = [
            (neighbor, similarity)
            for neighbor, similarity in neighbors
            if movie_id in self.user_ratings[neighbor]
        ]
        
        if not relevant_neighbors:
            # Return average rating of user if no neighbors rated the movie
            if self.user_ratings[user_id]:
                avg_rating = sum(self.user_ratings[user_id].values()) / len(self.user_ratings[user_id])
                return avg_rating, []
            return 0.0, []
        
        # Calculate weighted average
        numerator = sum(
            similarity * self.user_ratings[neighbor][movie_id]
            for neighbor, similarity in relevant_neighbors
        )
        
        denominator = sum(similarity for _, similarity in relevant_neighbors)
        
        if denominator == 0:
            return 0.0, relevant_neighbors
        
        predicted_rating = numerator / denominator
        return predicted_rating, relevant_neighbors
    
    def recommend_movies(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float, str]]:
        """
        Recommend top N movies for a user
        
        Returns:
            List of (movie_id, predicted_rating, movie_title) tuples
        """
        if user_id not in self.user_ratings:
            return []
        
        # Get movies user hasn't rated
        user_movies = set(self.user_ratings[user_id].keys())
        all_movies = set(self.movie_ratings.keys())
        unrated_movies = all_movies - user_movies
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating, neighbors = self.predict_rating(user_id, movie_id)
            if predicted_rating > 0:
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def evaluate(self, test_data: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """
        Evaluate recommender performance
        
        Args:
            test_data: List of (user_id, movie_id, actual_rating) tuples
        
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        actuals = []
        
        for user_id, movie_id, actual_rating in test_data:
            predicted_rating, _ = self.predict_rating(user_id, movie_id)
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        # Calculate metrics
        n = len(predictions)
        if n == 0:
            return {'mae': 0.0, 'rmse': 0.0, 'accuracy': 0.0}
        
        # Mean Absolute Error
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
        
        # Root Mean Squared Error
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n)
        
        # Accuracy (within 0.5 rating)
        correct = sum(1 for p, a in zip(predictions, actuals) if abs(p - a) <= 0.5)
        accuracy = correct / n
        
        return {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'n_predictions': n
        }


def generate_movie_data():
    """Generate synthetic movie ratings data"""
    import random
    random.seed(42)
    
    # Movie database
    movies = {
        1: "The Matrix", 2: "Inception", 3: "Interstellar", 4: "The Dark Knight",
        5: "Pulp Fiction", 6: "The Shawshank Redemption", 7: "Forrest Gump",
        8: "The Godfather", 9: "Fight Club", 10: "Goodfellas",
        11: "The Lord of the Rings", 12: "Star Wars", 13: "Jurassic Park",
        14: "Titanic", 15: "Avatar", 16: "The Avengers", 17: "Iron Man",
        18: "Spider-Man", 19: "Batman Begins", 20: "The Prestige"
    }
    
    # Generate user preferences
    user_preferences = {
        'User1': [1, 2, 3, 4, 19, 20],      # Sci-fi/Action fan
        'User2': [5, 6, 7, 8, 9, 10],       # Drama/Thriller fan
        'User3': [11, 12, 13, 16, 17, 18],  # Fantasy/Superhero fan
        'User4': [1, 2, 4, 16, 17, 19],     # Action/Superhero fan
        'User5': [5, 6, 7, 8, 14],          # Classic drama fan
        'User6': [11, 12, 13, 15],          # Fantasy/Epic fan
        'User7': [1, 3, 4, 9, 20],          # Mind-bending films fan
        'User8': [6, 7, 8, 10, 14],         # Oscar winners fan
    }
    
    # Generate ratings
    ratings_data = []
    for user_idx, (user_name, preferred_movies) in enumerate(user_preferences.items(), 1):
        for movie_id in preferred_movies:
            # Higher ratings for preferred genres
            rating = random.uniform(3.5, 5.0)
            ratings_data.append((user_idx, movie_id, rating, user_name))
        
        # Add some random ratings for other movies
        other_movies = set(movies.keys()) - set(preferred_movies)
        for movie_id in random.sample(list(other_movies), 3):
            rating = random.uniform(2.0, 4.0)
            ratings_data.append((user_idx, movie_id, rating, user_name))
    
    return ratings_data, movies, user_preferences


def main():
    """Main execution function"""
    print("=" * 70)
    print("K-Nearest Neighbors: Movie Recommendation System")
    print("=" * 70)
    print()
    
    # Generate data
    print("Step 1: Loading Movie Ratings Database")
    print("-" * 70)
    ratings_data, movies, user_preferences = generate_movie_data()
    print(f"Loaded {len(movies)} movies")
    print(f"Loaded {len(ratings_data)} ratings from 8 users")
    print()
    
    # Build recommender
    print("Step 2: Building KNN Recommender System")
    print("-" * 70)
    recommender = KNNRecommender(k=3, similarity_metric='cosine')
    
    # Split data into train and test
    train_data = ratings_data[:-20]
    test_data = ratings_data[-20:]
    
    # Add training data
    for user_id, movie_id, rating, _ in train_data:
        recommender.add_rating(user_id, movie_id, rating)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"K (neighbors): {recommender.k}")
    print(f"Similarity metric: {recommender.similarity_metric}")
    print()
    
    # Test different similarity metrics
    print("Step 3: Comparing Similarity Metrics")
    print("-" * 70)
    
    metrics_to_test = ['cosine', 'euclidean', 'pearson']
    for metric in metrics_to_test:
        recommender.similarity_metric = metric
        recommender.similarity_cache = {}  # Clear cache
        
        results = recommender.evaluate(test_data)
        print(f"{metric.capitalize()} Similarity:")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  Accuracy (±0.5): {results['accuracy']*100:.2f}%")
        print()
    
    # Reset to cosine
    recommender.similarity_metric = 'cosine'
    recommender.similarity_cache = {}
    
    # Generate recommendations
    print("Step 4: Generating Personalized Recommendations")
    print("-" * 70)
    
    test_users = [1, 3, 5]
    for user_id in test_users:
        print(f"\nRecommendations for User {user_id}:")
        
        # Show user's rated movies
        user_movies = recommender.user_ratings.get(user_id, {})
        print(f"  User has rated {len(user_movies)} movies")
        
        # Find neighbors
        neighbors = recommender.find_k_nearest_neighbors(user_id)
        print(f"  Nearest neighbors: ", end="")
        print(", ".join([f"User{n[0]} (sim: {n[1]:.3f})" for n in neighbors[:3]]))
        
        # Get recommendations
        recommendations = recommender.recommend_movies(user_id, n_recommendations=5)
        print(f"  Top 5 recommendations:")
        for i, (movie_id, predicted_rating) in enumerate(recommendations, 1):
            print(f"    {i}. {movies[movie_id]} (predicted: {predicted_rating:.2f}⭐)")
        print()
    
    # Analyze similarity matrix
    print("\nStep 5: User Similarity Analysis")
    print("-" * 70)
    print("User similarity matrix (cosine):")
    print()
    
    users = sorted(recommender.user_ratings.keys())[:6]
    print("    ", end="")
    for u in users:
        print(f"U{u}   ", end="")
    print()
    
    for u1 in users:
        print(f"U{u1}  ", end="")
        for u2 in users:
            if u1 == u2:
                print("1.00 ", end="")
            else:
                sim = recommender.calculate_similarity(u1, u2)
                print(f"{sim:.2f} ", end="")
        print()
    print()
    
    # Final summary
    print("\n" + "=" * 70)
    print("Recommendation System Summary")
    print("=" * 70)
    print(f"✓ Processed {len(ratings_data)} movie ratings")
    print(f"✓ Built KNN collaborative filtering model")
    print(f"✓ Tested multiple similarity metrics")
    print(f"✓ Generated personalized recommendations")
    print(f"✓ Achieved {results['accuracy']*100:.1f}% prediction accuracy")
    print()
    print("Key Insights:")
    print("- Similar users have similar movie preferences")
    print("- Cosine similarity works well for sparse rating matrices")
    print("- More neighbors don't always improve accuracy")
    print("- Cold start problem exists for new users/movies")
    print()

if __name__ == "__main__":
    main()

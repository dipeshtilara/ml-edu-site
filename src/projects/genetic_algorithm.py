"""
Genetic Algorithm: Feature Selection Optimization
Comprehensive implementation of genetic algorithm for feature selection
CBSE Class 12 AI Project
"""

import json
import random
import math
from typing import List, Tuple, Dict, Any, Callable

class GeneticAlgorithm:
    """
    Genetic Algorithm for optimization problems
    """
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.01,
                 crossover_rate: float = 0.7, elitism_count: int = 2):
        """
        Initialize genetic algorithm
        
        Args:
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_count: Number of best individuals to preserve
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.population = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.history = []
    
    def initialize_population(self, chromosome_length: int) -> List[List[int]]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            # Binary chromosome (0 or 1 for each gene)
            chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
            # Ensure at least one feature is selected
            if sum(chromosome) == 0:
                chromosome[random.randint(0, chromosome_length - 1)] = 1
            population.append(chromosome)
        return population
    
    def calculate_fitness(self, chromosome: List[int], 
                         fitness_function: Callable[[List[int]], float]) -> float:
        """Calculate fitness of a chromosome"""
        return fitness_function(chromosome)
    
    def select_parents(self, population: List[List[int]], 
                      fitnesses: List[float]) -> Tuple[List[int], List[int]]:
        """Select two parents using tournament selection"""
        tournament_size = 3
        
        def tournament_select():
            tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            return winner[0]
        
        parent1 = tournament_select()
        parent2 = tournament_select()
        return parent1, parent2
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Perform single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        length = len(parent1)
        crossover_point = random.randint(1, length - 1)
        
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Ensure at least one feature is selected
        if sum(offspring1) == 0:
            offspring1[random.randint(0, length - 1)] = 1
        if sum(offspring2) == 0:
            offspring2[random.randint(0, length - 1)] = 1
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Perform bit-flip mutation"""
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least one feature is selected
        if sum(mutated) == 0:
            mutated[random.randint(0, len(mutated) - 1)] = 1
        
        return mutated
    
    def evolve(self, fitness_function: Callable[[List[int]], float], 
              chromosome_length: int, generations: int = 100) -> Dict[str, Any]:
        """
        Run genetic algorithm
        
        Args:
            fitness_function: Function to evaluate chromosome fitness
            chromosome_length: Length of chromosome
            generations: Number of generations to evolve
        
        Returns:
            Dictionary containing best solution and history
        """
        # Initialize population
        self.population = self.initialize_population(chromosome_length)
        self.history = []
        
        for generation in range(generations):
            # Calculate fitness for all individuals
            fitnesses = [self.calculate_fitness(ind, fitness_function) 
                        for ind in self.population]
            
            # Track best individual
            max_fitness_idx = fitnesses.index(max(fitnesses))
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_individual = self.population[max_fitness_idx].copy()
            
            # Record history
            avg_fitness = sum(fitnesses) / len(fitnesses)
            self.history.append({
                'generation': generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': avg_fitness,
                'worst_fitness': min(fitnesses)
            })
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite_indices = sorted(range(len(fitnesses)), 
                                  key=lambda i: fitnesses[i], 
                                  reverse=True)[:self.elitism_count]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(self.population, fitnesses)
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            self.population = new_population
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'history': self.history
        }


def create_feature_selection_problem():
    """Create synthetic feature selection problem"""
    random.seed(42)
    
    # Generate synthetic dataset
    n_samples = 200
    n_features = 20
    
    # Important features: 0, 3, 7, 12, 15
    important_features = [0, 3, 7, 12, 15]
    
    # Generate data
    X = [[random.gauss(0, 1) for _ in range(n_features)] for _ in range(n_samples)]
    
    # Generate labels based on important features
    y = []
    for sample in X:
        value = sum(sample[i] for i in important_features)
        y.append(1 if value > 0 else 0)
    
    return X, y, important_features, n_features


def create_fitness_function(X: List[List[float]], y: List[int]):
    """Create fitness function for feature selection"""
    
    def fitness(chromosome: List[int]) -> float:
        # Penalize if no features selected
        if sum(chromosome) == 0:
            return 0.0
        
        # Simple accuracy calculation using selected features
        correct = 0
        selected_features = [i for i, bit in enumerate(chromosome) if bit == 1]
        
        for sample, label in zip(X, y):
            # Calculate prediction using selected features
            value = sum(sample[i] for i in selected_features)
            prediction = 1 if value > 0 else 0
            if prediction == label:
                correct += 1
        
        accuracy = correct / len(y)
        
        # Penalize for using too many features (encourage simplicity)
        feature_penalty = sum(chromosome) / len(chromosome) * 0.1
        
        return accuracy - feature_penalty
    
    return fitness


def main():
    """Main execution function"""
    print("=" * 70)
    print("Genetic Algorithm: Feature Selection Optimization")
    print("=" * 70)
    print()
    
    # Create problem
    print("Step 1: Creating Feature Selection Problem")
    print("-" * 70)
    X, y, true_important_features, n_features = create_feature_selection_problem()
    print(f"Dataset: {len(X)} samples, {n_features} features")
    print(f"True important features: {true_important_features}")
    print(f"Class distribution: {sum(y)} positive, {len(y) - sum(y)} negative")
    print()
    
    # Create fitness function
    fitness_function = create_fitness_function(X, y)
    
    # Initialize GA
    print("Step 2: Initializing Genetic Algorithm")
    print("-" * 70)
    ga = GeneticAlgorithm(
        population_size=50,
        mutation_rate=0.05,
        crossover_rate=0.7,
        elitism_count=2
    )
    print(f"Population size: {ga.population_size}")
    print(f"Mutation rate: {ga.mutation_rate}")
    print(f"Crossover rate: {ga.crossover_rate}")
    print(f"Elitism count: {ga.elitism_count}")
    print()
    
    # Run evolution
    print("Step 3: Evolving Population")
    print("-" * 70)
    generations = 100
    print(f"Running for {generations} generations...")
    result = ga.evolve(fitness_function, n_features, generations)
    print(f"Evolution completed!")
    print()
    
    # Show evolution progress
    print("Evolution Progress (every 10 generations):")
    for i in range(0, len(result['history']), 10):
        record = result['history'][i]
        print(f"  Gen {record['generation']:3d}: "
              f"Best={record['best_fitness']:.4f}, "
              f"Avg={record['avg_fitness']:.4f}")
    print()
    
    # Analyze best solution
    print("Step 4: Analyzing Best Solution")
    print("-" * 70)
    best_individual = result['best_individual']
    selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
    
    print(f"Best fitness: {result['best_fitness']:.4f}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"True important features: {true_important_features}")
    print()
    
    # Calculate overlap
    correct_features = set(selected_features) & set(true_important_features)
    precision = len(correct_features) / len(selected_features) if selected_features else 0
    recall = len(correct_features) / len(true_important_features)
    
    print("Feature Selection Quality:")
    print(f"  Correctly identified: {len(correct_features)}/{len(true_important_features)} important features")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print()
    
    # Show chromosome
    print("Best Chromosome (1=selected, 0=not selected):")
    print("  ", end="")
    for i, bit in enumerate(best_individual):
        marker = "*" if i in true_important_features else " "
        print(f"{bit}{marker}", end=" ")
    print()
    print("  (* = truly important feature)")
    print()
    
    # Summary
    print("\n" + "=" * 70)
    print("Genetic Algorithm Summary")
    print("=" * 70)
    print(f"✓ Evolved population for {generations} generations")
    print(f"✓ Found solution with fitness {result['best_fitness']:.4f}")
    print(f"✓ Selected {len(selected_features)} features")
    print(f"✓ Identified {len(correct_features)}/{len(true_important_features)} important features")
    print()
    print("Key Concepts:")
    print("• Genetic algorithms use evolution principles for optimization")
    print("• Selection, crossover, and mutation drive evolution")
    print("• Elitism preserves best solutions across generations")
    print("• Fitness function guides evolution toward better solutions")
    print("• Balance between exploration and exploitation is crucial")
    print()

if __name__ == "__main__":
    main()

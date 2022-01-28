import numpy as np
import random
import matplotlib.pyplot as plt

def generate_population(population_size: int, n_features: int):
    return np.random.randint(2, size=(population_size, n_features))

def sine_fitness_function(population: np.array, n_features: int):
    decimal_population = population.dot(1 << np.arange(population.shape[-1]))
    scaling_factor = n_features - 7 # 2^7 = 128 which is the max value
    normalized_population = decimal_population / (2**scaling_factor)
    assert np.amin(normalized_population) >= 0 and np.amax(normalized_population) <= 128  # Check interval
    sin_values = np.sin(normalized_population) # [-1, 1]
    return normalized_population, sin_values


# Roulette wheel selection
def roulette_selection(fitness: np.array, population: np.array, n: int):
    fitness = (fitness + 1) / 2 # [-1, 1] -> [0, 1] 
    probability = fitness/np.sum(fitness)
    indices = np.array([i for i in range(population.shape[0])])
    survivor_indices = np.random.choice(indices, n, p = probability)
    return population[survivor_indices]

def single_point_crossover(A, B, p):
    A_cross = np.append(A[:p], B[p:])
    B_cross = np.append(B[:p], A[p:])
    return A_cross, B_cross

def create_offspring(parents: np.array, n_offspring:int, p_c: int, p_m: int):
    assert parents.shape[0] % 2 == 0 # Has to be an even number
    assert parents.shape[0] >= n_offspring

    shuffled_indices = np.array([i for i in range(parents.shape[0])])
    np.random.shuffle(shuffled_indices)
    offspring = np.zeros((n_offspring, parents.shape[1]))

    # Crossover
    for i in range(0, n_offspring, 2):
        if random.random() < p_c:
            crossover_point = np.random.randint(low = 1, high = parents.shape[0])
            offspring[i], offspring[i+1] = single_point_crossover(parents[i], parents[i+1], crossover_point)
            
    # Mutation  
    for i in range(n_offspring):
        if random.random() < p_m:
            mutation_point = np.random.randint(offspring.shape[1])
            offspring[i][mutation_point] = 1 - offspring[i][mutation_point] # bit-flip
    
    return offspring

def survivor_selection_top(parents: np.array, offspring: np.array, survivor_size: int):
    normalized_parents, parents_fitness = sine_fitness_function(parents, genetic_size)
    normalized_offspring, offspring_fitness = sine_fitness_function(offspring, genetic_size)
    new_population = np.concatenate((parents, offspring), axis=0)
    new_offspring = np.concatenate((parents_fitness, offspring_fitness), axis=0)

    top_fitness_indices = np.argpartition(new_offspring, -survivor_size)[-survivor_size:]
    return new_population[top_fitness_indices]



if __name__ == "__main__":
    population_size = 50
    genetic_size = 30
    n_parents = population_size    
    n_offspring = population_size  
    crossover_rate = 1.0            # p_c 1.0 => two offsprings per parents
    mutation_rate = 0.85            # p_m
    n_generations = 10

    # a) Implement a function to generate an initial population for your genetic algorithm
    population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)

    for gen in range(n_generations):

        # b) Implement a parent selection function
        normalized, fitness = sine_fitness_function(population, genetic_size) # Normalized to range [0, 128]
        parents = roulette_selection(fitness, population, n_parents)
        fitness_array[gen] = np.average(fitness)

        # c) Crossover and mutation
        offspring = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        # d) Implement survivor selection
        # TODO Add another survivor selection function
        survivors = survivor_selection_top(parents, offspring, population_size)

        population = survivors

    print(f'Final fitness average {np.round(np.average(fitness)*100, 6)}%')
    print('Unique normalized x values:', np.unique(np.round(normalized)))

    plt.plot(fitness_array)
    plt.show()

    x = np.arange(0, 128, 0.1) 
    y = np.sin(x)
    plt.plot(x, y, color='blue')
    plt.plot(normalized, fitness, linestyle="", marker="o", color='red')
    plt.xlim(0, 128)
    plt.ylim(-1, 1)
    plt.show()
    plt.close()



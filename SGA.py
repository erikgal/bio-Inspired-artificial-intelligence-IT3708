import numpy as np
import random
import matplotlib.pyplot as plt
from LinReg import LinReg

def generate_population(population_size: int, n_features: int):
    return np.random.randint(2, size=(population_size, n_features))

def sine_fitness_function(population: np.array, n_features: int):
    decimal_population = population.dot(1 << np.arange(population.shape[-1])) # Binary to Decimal
    scaling_factor = n_features - 7 # 2^7 = 128 which is the max value
    normalized_population = decimal_population / (2**scaling_factor)
    assert np.amin(normalized_population) >= 0 and np.amax(normalized_population) <= 128  # Check interval
    sin_values = np.sin(normalized_population) # [-1, 1]
    return normalized_population, sin_values

def sine_penality_fitness_function(population: np.array, min_val: int, max_val: int):
    decimal_population = population.dot(1 << np.arange(population.shape[-1]))
    decimal_population[decimal_population < min_val] = min_val
    decimal_population[decimal_population > max_val] = max_val
    sin_values = np.sin(decimal_population) # [-1, 1]
    return decimal_population, sin_values

def roulette_selection(fitness: np.array, population: np.array, n: int):
    fitness = (fitness + 1) / 2 # [-1, 1] -> [0, 1] 
    probability = fitness/np.sum(fitness)
    indices = np.array([i for i in range(population.shape[0])])
    survivor_indices = np.random.choice(indices, n, p = probability)
    return population[survivor_indices]

def parent_selection_top(fitness: np.array, population: np.array, n: int):
    top_fitness_indices = np.argpartition(fitness, -n)[-n:]
    return population[top_fitness_indices]

def single_point_crossover(A, B, p):
    A_cross = np.append(A[:p], B[p:])
    B_cross = np.append(B[:p], A[p:])
    return A_cross, B_cross

def create_offspring(parents: np.array, n_offspring:int, p_c: int, p_m: int):
    offspring = np.zeros((n_offspring, parents.shape[1]), dtype=int)
    offspring_parents = np.zeros(n_offspring, dtype=int)

    # Crossover
    for i in range(0, n_offspring, 2):
        p1_index = np.random.randint(low = 0, high = parents.shape[0])
        p2_index = np.random.randint(low = 0, high = parents.shape[0])
        offspring_parents[i] = p1_index
        offspring_parents[i+1] = p2_index

        if random.random() < p_c:
            crossover_point = np.random.randint(low = 1, high = parents.shape[1])
            offspring[i], offspring[i+1] = single_point_crossover(parents[p1_index], parents[p2_index], crossover_point)
            
    # Mutation  
    for i in range(n_offspring):
        for j, bit in enumerate(offspring[i]):
            if random.random() < p_m:
                offspring[i][j] = 1 - bit # bit-flip

    return offspring, offspring_parents

def survivor_selection_top(parents: np.array, offspring: np.array, survivor_size: int, function):
    normalized_parents, parents_fitness = function(parents, genetic_size)
    normalized_offspring, offspring_fitness = function(offspring, genetic_size)
    new_population = np.concatenate((parents, offspring), axis=0)
    new_offspring = np.concatenate((parents_fitness, offspring_fitness), axis=0)

    top_fitness_indices = np.argpartition(new_offspring, -survivor_size)[-survivor_size:]
    return new_population[top_fitness_indices]

def ml_fitness(population: np.array, geneteic_size: int):
    fitness = np.zeros(population.shape[0])
    lin_reg = LinReg()
    x = data[:, :-1]
    y = data[:, -1]
    for i, bitstring in enumerate(population):
        bitstring = map(str, bitstring)
        filtered_x = lin_reg.get_columns(x, bitstring)
        fitness[i] = lin_reg.get_fitness(filtered_x, y)
    return x, 0.150 - fitness

def hamming_distance(bitstring1, bitstring2):
    return np.count_nonzero(bitstring1!=bitstring2)

def survivor_crowding_replacement(parents: np.array, offspring: np.array, offspring_parents: np.array,  survivor_size: int, fitness_function):
    survivors = np.zeros((n_offspring, parents.shape[1]), dtype=int)
    for i in range(0, offspring.shape[0], 2):
        o_1, o_2 = offspring[i], offspring[i+1]
        p_1, p_2 = parents[offspring_parents[i]], parents[offspring_parents[i+1]]
        o1_p1, o1_p2 = hamming_distance(o_1, p_1),  hamming_distance(o_1, p_2)
        o2_p2, o2_p1 = hamming_distance(o_2, p_2), hamming_distance(o_2, p_1)

        if o1_p1 + o2_p2 < o1_p2 + o2_p1:
            comp1, comp2 = np.array([o_1, p_1]), np.array([o_2, p_2])
        else:
            comp1, comp2 = np.array([o_1, p_2]), np.array([o_2, p_1])
        
        normalized1, fitness1 = fitness_function(comp1, genetic_size)
        normalized2, fitness2 = fitness_function(comp2, genetic_size)
        winner1 = comp1[np.argpartition(fitness1, -1)[-1:]]
        winner2 = comp2[np.argpartition(fitness2, -1)[-1:]]
        survivors[i], survivors[i+1] = winner1, winner2
    
    return survivors


if __name__ == "__main__":
    population_size = 50
    genetic_size = 30
    n_parents = population_size    
    n_offspring = population_size  
    crossover_rate = 1.0               # p_c 1.0 => two offsprings per parents
    mutation_rate = 1/population_size  # p_m
    n_generations = 100

    # a) Implement a function to generate an initial population for your genetic algorithm
    population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)

    for gen in range(n_generations):

        # b) Implement a parent selection function
        normalized, fitness = sine_fitness_function(population, genetic_size) # Normalized to range [0, 128]
        # parents = roulette_selection(fitness, population, n_parents) # Stochastic
        parents = parent_selection_top(fitness, population, n_parents) # Deterministic
        fitness_array[gen] = np.average(fitness)

        # c) Crossover and mutation
        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        # d) Implement survivor selection
        # TODO Add another survivor selection function
        survivors = survivor_selection_top(parents, offspring, population_size, sine_fitness_function) # Deterministic

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

    population_size = 200
    genetic_size = 4
    n_parents = population_size    
    n_offspring = population_size  
    crossover_rate = 1.0                  
    mutation_rate = 1/population_size           
    n_generations = 10

    # f) Add the constraint that the solution must reside in the interval [5, 10]
    population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)

    for gen in range(n_generations):

        normalized, fitness = sine_penality_fitness_function(population, 5, 10) # Penalize to [5, 10]
        parents = roulette_selection(fitness, population, n_parents) # Stochastic
        # parents = parent_selection_top(fitness, population, n_parents) # Deterministic
        fitness_array[gen] = np.average(fitness)

        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        survivors = survivor_selection_top(parents, offspring, population_size, sine_fitness_function) # Deterministic

        population = survivors

    print(f'Final fitness average {np.round(np.average(fitness)*100, 6)}%')
    print('Unique normalized x values:', np.unique(np.round(normalized)))

    plt.plot(fitness_array)
    plt.show()

    x = np.arange(5, 10, 0.1) 
    y = np.sin(x)
    plt.plot(x, y, color='blue')
    plt.plot(normalized, fitness, linestyle="", marker="o", color='red')
    plt.xlim(5, 10)
    plt.ylim(-1, 1)
    plt.show()
    plt.close()
    

    # g) Run the genetic algorithm on the provided dataset.

    # Read data from Dataset.txt
    file = open('Dataset.txt', 'r').read().split()
    data = np.zeros((len(file), len(file[0].split(','))))
    for i, line in enumerate(file):
        data[i] = line.split(',')

    population_size = 100
    genetic_size = data.shape[1]-1 # NB the label is not a feature
    n_parents = population_size    
    n_offspring = population_size  
    crossover_rate = 1.0               
    mutation_rate = 1/population_size       
    n_generations = 70

    population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)

    fitness_array = np.zeros(n_generations)

    for gen in range(n_generations):
        print('Gen:', gen)

        x, fitness = ml_fitness(population, genetic_size)
        # parents = roulette_selection(fitness, population, n_parents) # Stochastic
        parents = parent_selection_top(fitness, population, n_parents) # Deterministic
        fitness_array[gen] = np.average(0.150 - fitness)
        print('Average RMSE', np.average(0.150 - fitness))

        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        survivors = survivor_selection_top(parents, offspring, population_size, ml_fitness) # Deterministic

        population = survivors

    print(f'Final RMSE average {np.average(0.150 - fitness)}')
    print(f'Final best RMSE {0.150 - np.amax(fitness)}')


    plt.plot(fitness_array)
    plt.xlabel('Generation')
    plt.ylabel('Average RMSE') 
    plt.show()

    # h) Implement a new survivor selection function with crowding
    population_size = 500
    genetic_size = 30
    n_parents = population_size    
    n_offspring = population_size  
    crossover_rate = 1.0           
    mutation_rate = 1/population_size            
    n_generations = 50

    population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)

    for gen in range(n_generations):

        normalized, fitness = sine_fitness_function(population, genetic_size) # Normalized to range [0, 128]
        # parents = roulette_selection(fitness, population, n_parents) # Stochastic
        parents = parent_selection_top(fitness, population, n_parents) # Deterministic
        fitness_array[gen] = np.average(fitness)

        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        survivors = survivor_crowding_replacement(parents, offspring, offspring_parents, population_size, sine_fitness_function) # Deterministic

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

    # Read data from Dataset.txt
    file = open('Dataset.txt', 'r').read().split()
    data = np.zeros((len(file), len(file[0].split(','))))
    for i, line in enumerate(file):
        data[i] = line.split(',')

    population_size = 100
    genetic_size = data.shape[1]-1
    n_parents = population_size    
    n_offspring = population_size  
    crossover_rate = 1.0               
    mutation_rate = 1/population_size       
    n_generations = 70

    population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)

    fitness_array = np.zeros(n_generations)

    for gen in range(n_generations):
        print('Gen:', gen)

        x, fitness = ml_fitness(population, genetic_size)
        # parents = roulette_selection(fitness, population, n_parents) # Stochastic
        parents = parent_selection_top(fitness, population, n_parents) # Deterministic
        fitness_array[gen] = np.average(0.150 - fitness)
        print('Average RMSE', np.average(0.150 - fitness))

        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        survivors = survivor_crowding_replacement(parents, offspring, offspring_parents, population_size, ml_fitness) # Deterministic

        population = survivors

    print(f'Final RMSE average {np.average(0.150 - fitness)}')
    print(f'Final best RMSE {0.150 - np.amax(fitness)}')


    plt.plot(fitness_array)
    plt.xlabel('Generation')
    plt.ylabel('Average RMSE') 
    plt.show()


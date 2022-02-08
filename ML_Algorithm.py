import numpy as np
import random
import matplotlib.pyplot as plt
from LinReg import LinReg

def generate_population(population_size: int, n_features: int):
    return np.random.randint(2, size=(population_size, n_features))

def parent_selection_top(fitness: np.ndarray, population: np.ndarray, n: int):
    top_fitness_indices = np.argpartition(fitness, -n)[-n:]
    return population[top_fitness_indices]

def single_point_crossover(A, B, p):
    A_cross = np.append(A[:p], B[p:])
    B_cross = np.append(B[:p], A[p:])
    return A_cross, B_cross

def create_offspring(parents: np.ndarray, n_offspring:int, p_c: int, p_m: int):
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
        else:
            offspring[p1_index], offspring[p2_index] 

    # Mutation  
    for i in range(n_offspring):
        for j, bit in enumerate(offspring[i]):
            if random.random() < p_m:
                offspring[i][j] = 1 - bit # bit-flip

    return offspring, offspring_parents

def ml_fitness(population: np.ndarray):
    fitness = np.zeros(population.shape[0])
    lin_reg = LinReg()
    x = data[:, :-1]
    y = data[:, -1]
    for i, bitstring in enumerate(population):
        bitstring = map(str, bitstring)
        filtered_x = lin_reg.get_columns(x, bitstring)
        fitness[i] = lin_reg.get_fitness(filtered_x, y)
    return 0.150 - fitness

def survivor_selection_top(parents: np.ndarray, offspring: np.ndarray, survivor_size: int, function):
    parents_fitness = function(parents)
    offspring_fitness = function(offspring)
    new_population = np.concatenate((parents, offspring), axis=0)
    new_offspring = np.concatenate((parents_fitness, offspring_fitness), axis=0)

    top_fitness_indices = np.argpartition(new_offspring, -survivor_size)[-survivor_size:]
    return new_population[top_fitness_indices]

def hamming_distance(bitstring1, bitstring2):
    return np.count_nonzero(bitstring1!=bitstring2)

def survivor_crowding_replacement(parents: np.ndarray, offspring: np.ndarray, offspring_parents: np.ndarray, fitness_function):
    survivors = np.zeros((offspring.shape[0], parents.shape[1]), dtype=int)
    for i in range(0, offspring.shape[0], 2):
        o_1, o_2 = offspring[i], offspring[i+1]
        p_1, p_2 = parents[offspring_parents[i]], parents[offspring_parents[i+1]]
        o1_p1, o1_p2 = hamming_distance(o_1, p_1),  hamming_distance(o_1, p_2)
        o2_p2, o2_p1 = hamming_distance(o_2, p_2), hamming_distance(o_2, p_1)

        if o1_p1 + o2_p2 < o1_p2 + o2_p1:
            comp1, comp2 = np.array([o_1, p_1]), np.array([o_2, p_2])
        else:
            comp1, comp2 = np.array([o_1, p_2]), np.array([o_2, p_1])
        
        fitness1 = fitness_function(comp1)
        fitness2 = fitness_function(comp2)
        winner1 = comp1[np.argpartition(fitness1, -1)[-1:]]
        winner2 = comp2[np.argpartition(fitness2, -1)[-1:]]
        survivors[i], survivors[i+1] = winner1, winner2
    
    return survivors

def calculate_entropy(population: np.ndarray):
    entropy_sum = 0
    for i in range(population.shape[1]):
        bit_array = population[:, i]
        p_i = np.count_nonzero(bit_array == 1)/population.shape[0]
        if p_i > 0:
            entropy_sum += -p_i * np.log2(p_i)
    return entropy_sum

if __name__ == "__main__":
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
    n_generations = 100

    SGA_population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)
    SGA_entropy = np.zeros(n_generations)

    for gen in range(n_generations):
        print('Gen:', gen)

        fitness = ml_fitness(SGA_population)
        # parents = roulette_selection(fitness, population, n_parents) # Stochastic
        parents = parent_selection_top(fitness, SGA_population, n_parents) # Deterministic
        fitness_array[gen] = np.average(0.150 - fitness)
        SGA_entropy[gen] = calculate_entropy(SGA_population)
        print('Average RMSE', np.average(0.150 - fitness))

        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        survivors = survivor_selection_top(parents, offspring, population_size, ml_fitness) # Deterministic

        SGA_population = survivors

    print(f'Final RMSE average {np.average(0.150 - fitness)}')
    print(f'Final best RMSE {0.150 - np.amax(fitness)}')
    print(f'Results of not using any feature selection, RMSE: {LinReg().get_fitness(data[:, :-1], data[:, -1])}')

    plt.plot(fitness_array)
    plt.xlabel('Generation')
    plt.ylabel('Average RMSE') 
    plt.show()

    # h) Implement a new survivor selection function with crowding

    crowding_population = generate_population(population_size, genetic_size)
    fitness_array = np.zeros(n_generations)
    crowding_entropy = np.zeros(n_generations)

    for gen in range(n_generations):
        print('Gen:', gen)

        fitness = ml_fitness(crowding_population)
        # parents = roulette_selection(fitness, population, n_parents) # Stochastic
        parents = parent_selection_top(fitness, crowding_population, n_parents) # Deterministic
        fitness_array[gen] = np.average(0.150 - fitness)
        crowding_entropy[gen] = calculate_entropy(crowding_population)
        print('Average RMSE', np.average(0.150 - fitness))

        offspring, offspring_parents = create_offspring(parents, n_offspring, crossover_rate, mutation_rate)

        survivors = survivor_crowding_replacement(parents, offspring, offspring_parents, ml_fitness) # Deterministic

        crowding_population = survivors

    print(f'Final RMSE average: {np.average(0.150 - fitness)}')
    print(f'Final best RMSE: {0.150 - np.amax(fitness)}')

    plt.plot(fitness_array)
    plt.xlabel('Generation')
    plt.ylabel('Average RMSE') 
    plt.show()

    plt.plot(SGA_entropy)
    plt.plot(crowding_entropy)
    plt.legend(['SGA', 'Crowding'])
    plt.xlabel('Generation')
    plt.ylabel('Entropy')
    plt.show()
import numpy
import matplotlib.pyplot
import math
import random # Added for GA operations

# --- VRP Instance Data (Example) ---
# Define Bins: (id, location_x, location_y, trash_volume, service_time)
bins_data = {
    'bin1': {'loc': (2, 3), 'volume': 10, 'service_time': 0.1},
    'bin2': {'loc': (5, 6), 'volume': 8, 'service_time': 0.08},
    'bin3': {'loc': (1, 8), 'volume': 12, 'service_time': 0.12},
    'bin4': {'loc': (8, 2), 'volume': 15, 'service_time': 0.15},
    'bin5': {'loc': (6, 9), 'volume': 7, 'service_time': 0.07},
    'bin6': {'loc': (4, 1), 'volume': 9, 'service_time': 0.09},
    'bin7': {'loc': (9, 5), 'volume': 11, 'service_time': 0.11},
    'bin8': {'loc': (3, 7), 'volume': 13, 'service_time': 0.13},
}

totalTrash = 0
for bin_name in bins_data:
    bins_data[bin_name]['volume'] += 0
    totalTrash += bins_data[bin_name]['volume']
# Define Trucks
num_trucks = 2
truck_capacity = 50
truck_speed = 40  # units/hour
start_depot_location = (0, 0)
end_depot_location = (0, 0) # Assuming trucks return to the same depot

# Define Incinerator
incinerator_location = (10, 10)
incinerator_unload_time = 0.5  # hours
incinerator_id = 'INC'
depot_id = 'depot'

# --- Utility Function ---
def calculate_distance(loc1, loc2):
    """Calculates Euclidean distance between two locations."""
    return numpy.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# --- Core GA Functions for VRP ---

def calculate_fitness_vrp_simplified(chromosome, bins_data, truck_capacity, incinerator_location,
                                     start_depot_location, end_depot_location, truck_speed,
                                     incinerator_unload_time, weights):
    """
    Calculates the fitness of a single VRP chromosome (simplified: no area overlap).
    Lower fitness is better.
    """
    total_distance = 0
    total_time = 0
    total_incinerator_trips = 0

    all_bins_in_problem = set(bins_data.keys())
    visited_bins_in_chromosome = set()

    capacity_violation_penalty = 0
    unvisited_bin_penalty = 0
    invalid_bin_penalty_val = 0


    for truck_idx, truck_route in enumerate(chromosome):
        current_load = 0
        current_truck_distance = 0
        current_truck_time = 0
        last_location = start_depot_location

        if not truck_route or truck_route == [depot_id, depot_id] or truck_route == [depot_id]: # Skip empty or trivial routes
            continue
        
        # Ensure route starts and ends with depot_id for calculation if not already.
        # This is a simple fix; robust generation should ensure this.
        processed_route = list(truck_route) # Make a copy
        if not processed_route or processed_route[0] != depot_id:
            processed_route.insert(0, depot_id)
        if processed_route[-1] != depot_id:
            processed_route.append(depot_id)


        for i, stop_id in enumerate(processed_route):
            current_location = None
            if stop_id == depot_id:
                if i == 0: # Start of route
                    current_location = start_depot_location
                else: # End of route or intermediate depot visit
                    current_location = end_depot_location
            elif stop_id == incinerator_id:
                current_location = incinerator_location
                total_incinerator_trips += 1
                current_truck_time += incinerator_unload_time
                current_load = 0  # Emptied
            else:  # It's a bin
                if stop_id not in bins_data:
                    invalid_bin_penalty_val += weights.get('invalid_bin', 100000)
                    continue # Skip this unknown stop

                bin_info = bins_data[stop_id]
                current_location = bin_info['loc']
                visited_bins_in_chromosome.add(stop_id)

                # Check capacity BEFORE collecting (simplified check)
                # More advanced: check if *this specific collection* violates,
                # or if the route *should have had* an incinerator trip earlier.
                if current_load + bin_info['volume'] > truck_capacity:
                    # This implies a route construction flaw if an incinerator trip wasn't planned before this.
                    # For fitness, penalize.
                    capacity_violation_penalty += weights.get('capacity_violation', 10000) * \
                                                 (current_load + bin_info['volume'] - truck_capacity)
                
                current_load += bin_info['volume']
                current_truck_time += bin_info['service_time']

            if current_location is not None: # Ensure location was determined
                dist = calculate_distance(last_location, current_location)
                current_truck_distance += dist
                current_truck_time += dist / truck_speed if truck_speed > 0 else float('inf')
                last_location = current_location
            elif i > 0: # If current_location is None but it's not the first stop (problematic stop_id)
                pass


        total_distance += current_truck_distance
        total_time += current_truck_time

    unvisited_bins = all_bins_in_problem - visited_bins_in_chromosome
    if unvisited_bins:
        unvisited_bin_penalty = len(unvisited_bins) * weights.get('unvisited', 50000)

    fitness = (total_distance * weights.get('distance', 1.0) +
               total_time * weights.get('time', 1.0) +
               total_incinerator_trips * weights.get('incinerator_trips', 0.1) +
               unvisited_bin_penalty +
               capacity_violation_penalty +
               invalid_bin_penalty_val)
    return fitness


def create_initial_vrp_population(sol_per_pop, bins_data_keys, num_trucks, depot_id, incinerator_id):
    """
    Creates an initial population for the VRP.
    VERY BASIC AND LIKELY INEFFICIENT/INVALID INITIALIZATION.
    This needs significant improvement for real VRPs.
    """
    population = []
    all_bin_ids = list(bins_data_keys)

    for _ in range(sol_per_pop):
        chromosome = []
        shuffled_bins = list(all_bin_ids) # Make a mutable copy
        random.shuffle(shuffled_bins)

        # Distribute bins somewhat evenly (very naive)
        bins_per_truck_approx = len(shuffled_bins) // num_trucks if num_trucks > 0 else len(shuffled_bins)

        for i in range(num_trucks):
            route = [depot_id]
            truck_bins = []
            if num_trucks > 0 :
                start_idx = i * bins_per_truck_approx
                if i == num_trucks - 1: # Last truck takes all remaining
                    truck_bins = shuffled_bins[start_idx:]
                else:
                    truck_bins = shuffled_bins[start_idx : start_idx + bins_per_truck_approx]
            
            # Simple strategy: add bins, if capacity might be an issue, add INC (very naive)
            current_cap_sim = 0
            for bin_id in truck_bins:
                route.append(bin_id)
                current_cap_sim += bins_data[bin_id]['volume']
                if current_cap_sim > truck_capacity * 0.7 and random.random() < 0.3: # Arbitrary incinerator add
                    route.append(incinerator_id)
                    current_cap_sim = 0
            route.append(depot_id)
            chromosome.append(route)
        
        # If some trucks got no bins and others have too many, this initial pop is poor.
        # A repair or better generation heuristic is needed.
        population.append(chromosome)
    return population


def calculate_fitness_for_population(population, bins_data, truck_capacity, incinerator_loc,
                                     start_depot_loc, end_depot_loc, speed, unload_time, fitness_weights):
    fitness_scores = []
    for chromo in population:
        fitness_scores.append(calculate_fitness_vrp_simplified(
            chromo, bins_data, truck_capacity, incinerator_loc,
            start_depot_loc, end_depot_loc, speed,
            unload_time, fitness_weights
        ))
    return numpy.array(fitness_scores)


def select_mating_pool_vrp(population, fitness, num_parents):
    """Selects the best individuals in the current generation as parents for producing the next generation."""
    parents = []
    # Sort by fitness (lower is better because we minimize)
    sorted_indices = numpy.argsort(fitness)
    for i in range(num_parents):
        parents.append(population[sorted_indices[i]])
    return parents


def crossover_vrp(parents, offspring_size_tuple, bins_data_keys, depot_id, incinerator_id):
    """
    Performs crossover on parents to create offspring.
    VERY BASIC PLACEHOLDER - Often produces invalid or poor solutions.
    Needs a proper VRP crossover operator (e.g., Order Crossover, Route-based Crossover, PMX).
    """
    offspring = []
    num_offspring_needed = offspring_size_tuple[0]
    
    if not parents: return []

    for k in range(num_offspring_needed):
        # Simple: take routes from two random parents (can lead to bin duplication/omission)
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        
        # Example: Single point crossover on the list of routes
        # This is problematic as bin assignments are not globally managed.
        num_routes = len(parent1)
        child = [[] for _ in range(num_routes)]
        if num_routes > 0:
            crossover_point = random.randint(1, num_routes -1) if num_routes > 1 else 1
            child[:crossover_point] = [list(r) for r in parent1[:crossover_point]] # Deep copy routes
            child[crossover_point:] = [list(r) for r in parent2[crossover_point:]] # Deep copy routes
        
        # !!! CRITICAL FLAW: This crossover does NOT ensure all bins are visited exactly once
        # or that capacities are respected. A REPAIR FUNCTION IS ESSENTIAL HERE.
        # For this placeholder, we just return the potentially flawed child.
        offspring.append(child)
    return offspring


def mutation_vrp(offspring_crossover, mutation_rate, bins_data_keys, depot_id, incinerator_id):
    """
    Performs mutation on offspring.
    VERY BASIC PLACEHOLDER. Needs proper VRP mutation operators (swap, insert, invert).
    """
    mutated_offspring = []
    all_bin_ids = list(bins_data_keys)

    for chromosome in offspring_crossover:
        mutated_chromosome = [list(route) for route in chromosome] # Deep copy

        if random.random() < mutation_rate:
            # Example: Swap two bins within a random truck's route (very simple)
            if not mutated_chromosome: continue
            
            truck_idx_to_mutate = random.randrange(len(mutated_chromosome))
            route_to_mutate = mutated_chromosome[truck_idx_to_mutate]
            
            # Get indices of actual bins (not depot/incinerator)
            bin_indices_in_route = [i for i, stop in enumerate(route_to_mutate) 
                                    if stop != depot_id and stop != incinerator_id]

            if len(bin_indices_in_route) >= 2:
                idx1_map, idx2_map = random.sample(bin_indices_in_route, 2)
                route_to_mutate[idx1_map], route_to_mutate[idx2_map] = route_to_mutate[idx2_map], route_to_mutate[idx1_map]
        
        mutated_offspring.append(mutated_chromosome)
    return mutated_offspring

# --- GA Parameters ---
sol_per_pop = 50  # Population size
num_parents_mating = 16 # Number of parents to select
num_generations = 50   # Number of generations
mutation_rate = 0.25    # Mutation rate

# Fitness weights (prioritizing unvisited bins heavily)
fitness_weights_simplified = {
    'distance': 1.0,
    'time': 0.5,
    'unvisited': 100000, # Very high penalty for not visiting all bins
    'incinerator_trips': 10.0, # Penalize incinerator trips more to encourage efficiency
    'capacity_violation': 50000,
    'invalid_bin': 200000 # Penalty for non-existent bin ID in chromosome
}

# --- Main GA Loop ---
print("Starting Genetic Algorithm for VRP...")
print(f"Number of bins: {len(bins_data)}")
print(f"Number of trucks: {num_trucks}")

bin_ids_list = list(bins_data.keys()) # Pass only keys for generation/mutation

print("\nCreating initial VRP population...")
new_population = create_initial_vrp_population(sol_per_pop, bin_ids_list, num_trucks, depot_id, incinerator_id)

best_fitness_per_gen = []

for generation in range(num_generations):
    print(f"\n--- Generation : {generation} ---")

    fitness = calculate_fitness_for_population(
        new_population, bins_data, truck_capacity,
        incinerator_location, start_depot_location,
        end_depot_location, truck_speed,
        incinerator_unload_time, fitness_weights_simplified
    )

    current_best_fitness = numpy.min(fitness)
    best_fitness_per_gen.append(current_best_fitness)
    print(f"Best fitness in generation {generation}: {current_best_fitness:.2f}")
    # print(f"Worst fitness in generation {generation}: {numpy.max(fitness):.2f}")
    # print(f"Average fitness: {numpy.mean(fitness):.2f}")


    parents = select_mating_pool_vrp(new_population, fitness, num_parents_mating)

    offspring_crossover = crossover_vrp(parents, (sol_per_pop - len(parents),), bin_ids_list, depot_id, incinerator_id)

    offspring_mutation = mutation_vrp(offspring_crossover, mutation_rate, bin_ids_list, depot_id, incinerator_id)
    
    # Create the new population based on the parents and offspring
    # Elitism: keep the best parents
    new_population[:len(parents)] = parents 
    new_population[len(parents):] = offspring_mutation
    
    # Sanity check: ensure population size is maintained
    if len(new_population) != sol_per_pop:
        print(f"Warning: Population size changed to {len(new_population)}, expected {sol_per_pop}")
        # Basic fix: if too small, duplicate last element; if too large, truncate
        if len(new_population) < sol_per_pop:
            while len(new_population) < sol_per_pop: new_population.append(new_population[-1]) # duplicate
        else:
            new_population = new_population[:sol_per_pop]


# --- Results --- (Ensure this section starts after the GA loop)
print("\n--- GA Finished ---")
final_fitness_scores = calculate_fitness_for_population( # Renamed to avoid conflict
    new_population, bins_data, truck_capacity,
    incinerator_location, start_depot_location,
    end_depot_location, truck_speed,
    incinerator_unload_time, fitness_weights_simplified
)
best_match_idx = numpy.argmin(final_fitness_scores)

print("\nBest solution chromosome found: ")
best_route_overall = new_population[best_match_idx] # Get the best chromosome
for i, route_part in enumerate(best_route_overall):
    print(f"  Truck {i+1}: {route_part}")
print("Best solution fitness: ", final_fitness_scores[best_match_idx])

# --- Corrected Incinerator Trip Calculation for Display ---
print("\n--- Incinerator Trip Analysis for Best Solution ---")
minIncineratorTrips_theoretical = math.ceil(totalTrash / truck_capacity)
print(f"Total Trash: {totalTrash}")
print(f"Truck Capacity (per truck): {truck_capacity}")
print(f"Theoretical Minimum Incinerator Trips (overall): {minIncineratorTrips_theoretical}")

# Calculate trips for display, mirroring fitness function logic
evaluated_trips_in_best_solution = 0
for truck_route_chromosome_part in best_route_overall:
    current_load_sim = 0
    truck_serviced_any_bins = False
    
    # Count explicit INC trips and determine final load from chromosome part
    for stop_id in truck_route_chromosome_part:
        if stop_id == incinerator_id:
            evaluated_trips_in_best_solution += 1
            current_load_sim = 0
        elif stop_id != depot_id and stop_id in bins_data: # It's a bin
            current_load_sim += bins_data[stop_id]['volume']
            truck_serviced_any_bins = True
            # No depot handling needed here as we only care about load and INC for this count

    # Check for mandatory final trip (simulated by fitness function)
    if truck_serviced_any_bins and current_load_sim > 0:
        evaluated_trips_in_best_solution += 1 # This accounts for the mandatory final trip

print(f"Evaluated Incinerator Trips in Best Solution (explicit + mandatory final): {evaluated_trips_in_best_solution}")
# --- End Corrected Incinerator Trip Calculation ---

# --- Plotting ---
matplotlib.pyplot.plot(best_fitness_per_gen)
matplotlib.pyplot.xlabel("Generation")
matplotlib.pyplot.ylabel("Best Fitness (Lower is Better)")
matplotlib.pyplot.title("VRP Fitness Improvement Over Generations (Simplified)")
matplotlib.pyplot.grid(True)
matplotlib.pyplot.show()
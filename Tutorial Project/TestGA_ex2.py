import numpy
import matplotlib.pyplot as plt # Corrected import
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
    bins_data[bin_name]['volume'] += 10 # User modification
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

# --- Utility Functions ---
def calculate_distance(loc1, loc2):
    """Calculates Euclidean distance between two locations."""
    return numpy.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def get_location_coords(stop_id, bins_data_dict, depot_loc, incinerator_loc_coords, 
                        depot_id_str, incinerator_id_str):
    """Returns the (x, y) coordinates for a given stop ID."""
    if stop_id == depot_id_str:
        return depot_loc
    elif stop_id == incinerator_id_str:
        return incinerator_loc_coords
    elif stop_id in bins_data_dict:
        return bins_data_dict[stop_id]['loc']
    else:
        print(f"Warning: Location for stop_id '{stop_id}' not found in get_location_coords.")
        return None 

# --- Core GA Functions for VRP ---

def calculate_fitness_vrp_simplified(chromosome, bins_data, truck_capacity, incinerator_location,
                                     start_depot_location, end_depot_location, truck_speed,
                                     incinerator_unload_time, weights):
    """
    Calculates the fitness of a single VRP chromosome.
    This version reflects the user's provided script without the mandatory final incinerator trip logic
    within this function itself (that was a previous discussion point).
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

        if not truck_route or truck_route == [depot_id, depot_id] or truck_route == [depot_id]:
            continue
        
        processed_route = list(truck_route)
        if not processed_route or processed_route[0] != depot_id:
            processed_route.insert(0, depot_id)
        if processed_route[-1] != depot_id:
            processed_route.append(depot_id)

        for i, stop_id in enumerate(processed_route):
            current_location = None
            if stop_id == depot_id:
                if i == 0: 
                    current_location = start_depot_location
                else: 
                    current_location = end_depot_location
            elif stop_id == incinerator_id:
                current_location = incinerator_location
                total_incinerator_trips += 1
                current_truck_time += incinerator_unload_time
                current_load = 0 
            else: 
                if stop_id not in bins_data:
                    invalid_bin_penalty_val += weights.get('invalid_bin', 100000)
                    continue

                bin_info = bins_data[stop_id]
                current_location = bin_info['loc']
                visited_bins_in_chromosome.add(stop_id)

                if current_load + bin_info['volume'] > truck_capacity:
                    capacity_violation_penalty += weights.get('capacity_violation', 10000) * \
                                                 (current_load + bin_info['volume'] - truck_capacity)
                
                current_load += bin_info['volume']
                current_truck_time += bin_info['service_time']

            if current_location is not None:
                dist = calculate_distance(last_location, current_location)
                current_truck_distance += dist
                current_truck_time += dist / truck_speed if truck_speed > 0 else float('inf')
                last_location = current_location
            elif i > 0:
                pass

        total_distance += current_truck_distance
        total_time += current_truck_time

    unvisited_bins = all_bins_in_problem - visited_bins_in_chromosome
    if unvisited_bins:
        unvisited_bin_penalty = len(unvisited_bins) * weights.get('unvisited', 50000)

    fitness = (total_distance * weights.get('distance', 1.0) +
               total_time * weights.get('time', 1.0) +
               total_incinerator_trips * weights.get('incinerator_trips', 0.1) + # User's script has 0.1 here
               unvisited_bin_penalty +
               capacity_violation_penalty +
               invalid_bin_penalty_val)
    return fitness


def create_initial_vrp_population(sol_per_pop, bins_data_keys, num_trucks, depot_id, incinerator_id):
    population = []
    all_bin_ids = list(bins_data_keys)
    for _ in range(sol_per_pop):
        chromosome = []
        shuffled_bins = list(all_bin_ids)
        random.shuffle(shuffled_bins)
        bins_per_truck_approx = len(shuffled_bins) // num_trucks if num_trucks > 0 else len(shuffled_bins)
        for i in range(num_trucks):
            route = [depot_id]
            truck_bins = []
            if num_trucks > 0 :
                start_idx = i * bins_per_truck_approx
                if i == num_trucks - 1:
                    truck_bins = shuffled_bins[start_idx:]
                else:
                    truck_bins = shuffled_bins[start_idx : start_idx + bins_per_truck_approx]
            current_cap_sim = 0
            for bin_id in truck_bins:
                route.append(bin_id)
                if bin_id in bins_data: # Check if bin_id is valid
                    current_cap_sim += bins_data[bin_id]['volume']
                    if current_cap_sim > truck_capacity * 0.7 and random.random() < 0.3:
                        route.append(incinerator_id)
                        current_cap_sim = 0
            route.append(depot_id)
            chromosome.append(route)
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
    parents = []
    sorted_indices = numpy.argsort(fitness)
    for i in range(num_parents):
        parents.append(population[sorted_indices[i]])
    return parents


def crossover_vrp(parents, offspring_size_tuple, bins_data_keys, depot_id, incinerator_id):
    offspring = []
    num_offspring_needed = offspring_size_tuple[0]
    if not parents: return []
    for k in range(num_offspring_needed):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        num_routes = len(parent1)
        child = [[] for _ in range(num_routes)]
        if num_routes > 0 and len(parent2) == num_routes : # Basic check
            crossover_point = random.randint(1, num_routes -1) if num_routes > 1 else 1
            child[:crossover_point] = [list(r) for r in parent1[:crossover_point]]
            child[crossover_point:] = [list(r) for r in parent2[crossover_point:]]
        else: # Fallback if parent structures differ
             child = [list(r) for r in parent1] # Just copy parent1
        offspring.append(child)
    return offspring


def mutation_vrp(offspring_crossover, mutation_rate, bins_data_keys, depot_id, incinerator_id):
    mutated_offspring = []
    for chromosome in offspring_crossover:
        mutated_chromosome = [list(route) for route in chromosome]
        if random.random() < mutation_rate:
            if not mutated_chromosome: continue
            truck_idx_to_mutate = random.randrange(len(mutated_chromosome))
            route_to_mutate = mutated_chromosome[truck_idx_to_mutate]
            bin_indices_in_route = [i for i, stop in enumerate(route_to_mutate) 
                                    if stop != depot_id and stop != incinerator_id]
            if len(bin_indices_in_route) >= 2:
                idx1_map, idx2_map = random.sample(bin_indices_in_route, 2)
                route_to_mutate[idx1_map], route_to_mutate[idx2_map] = route_to_mutate[idx2_map], route_to_mutate[idx1_map]
        mutated_offspring.append(mutated_chromosome)
    return mutated_offspring

# --- Plotting Function for Routes ---
def plot_truck_routes(solution_chromosome, bins_data_dict, depot_loc, incinerator_loc_coords, 
                      depot_id_str, incinerator_id_str, fig_num=1):
    """Plots the truck routes from a solution chromosome."""
    plt.figure(fig_num, figsize=(12, 10))
    plt.clf() 

    # Plot Depot
    plt.plot(depot_loc[0], depot_loc[1], 'ks', markersize=10, label='Depot')
    plt.text(depot_loc[0], depot_loc[1] + 0.2, 'Depot', ha='center', va='bottom', fontsize=9)

    # Plot Incinerator
    plt.plot(incinerator_loc_coords[0], incinerator_loc_coords[1], 'm^', markersize=10, label='Incinerator')
    plt.text(incinerator_loc_coords[0], incinerator_loc_coords[1] + 0.2, 'Incinerator', ha='center', va='bottom', fontsize=9)

    # Plot Bins
    bin_label_added = False
    for bin_id, data in bins_data_dict.items():
        loc = data['loc']
        if not bin_label_added:
            plt.plot(loc[0], loc[1], 'bo', markersize=7, label='Bin')
            bin_label_added = True
        else:
            plt.plot(loc[0], loc[1], 'bo', markersize=7)
        plt.text(loc[0], loc[1] + 0.2, bin_id, ha='center', va='bottom', fontsize=8)

    route_colors = ['r', 'g', 'c', 'y', 'orange', 'purple', 'brown', 'pink'] # More colors

    for truck_idx, route in enumerate(solution_chromosome):
        if not route:
            continue

        current_path_segment_coords = []
        for stop_id in route: # The route from chromosome should already be complete
            coords = get_location_coords(stop_id, bins_data_dict, depot_loc, 
                                         incinerator_loc_coords, depot_id_str, incinerator_id_str)
            if coords:
                current_path_segment_coords.append(coords)
            
        if not current_path_segment_coords:
            continue

        path_x, path_y = zip(*current_path_segment_coords)
        
        color = route_colors[truck_idx % len(route_colors)]
        plt.plot(path_x, path_y, linestyle='-', color=color, marker='.', 
                 label=f'Truck {truck_idx+1} Route', alpha=0.8, linewidth=1.5)
        
        for i in range(len(path_x) - 1):
            plt.arrow(path_x[i], path_y[i], 
                      (path_x[i+1] - path_x[i]) * 0.95, # Shorten arrow slightly to not overlap marker
                      (path_y[i+1] - path_y[i]) * 0.95,
                      color=color, shape='full', lw=0, 
                      length_includes_head=True, head_width=0.15, alpha=0.6)

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Truck Routes for Best Solution (Explicit Chromosome Path)")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1)) # Legend outside
    
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend


# --- GA Parameters ---
sol_per_pop = 50
num_parents_mating = 16
num_generations = 50
mutation_rate = 0.25

fitness_weights_simplified = {
    'distance': 1.0,
    'time': 0.5,
    'unvisited': 100000,
    'incinerator_trips': 10.0, # User's script example output used 10.0 for this weight in analysis
    'capacity_violation': 50000,
    'invalid_bin': 200000
}

# --- Main GA Loop ---
print("Starting Genetic Algorithm for VRP...")
print(f"Number of bins: {len(bins_data)}")
print(f"Number of trucks: {num_trucks}")

bin_ids_list = list(bins_data.keys())

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

    parents = select_mating_pool_vrp(new_population, fitness, num_parents_mating)
    offspring_crossover = crossover_vrp(parents, (sol_per_pop - len(parents),), bin_ids_list, depot_id, incinerator_id)
    offspring_mutation = mutation_vrp(offspring_crossover, mutation_rate, bin_ids_list, depot_id, incinerator_id)
    
    new_population[:len(parents)] = parents 
    new_population[len(parents):] = offspring_mutation
    
    if len(new_population) != sol_per_pop:
        print(f"Warning: Population size changed to {len(new_population)}, expected {sol_per_pop}")
        if len(new_population) < sol_per_pop:
            while len(new_population) < sol_per_pop: new_population.append(new_population[-1])
        else:
            new_population = new_population[:sol_per_pop]

# --- Results ---
print("\n--- GA Finished ---")
final_fitness_scores = calculate_fitness_for_population(
    new_population, bins_data, truck_capacity,
    incinerator_location, start_depot_location,
    end_depot_location, truck_speed,
    incinerator_unload_time, fitness_weights_simplified
)
best_match_idx = numpy.argmin(final_fitness_scores)

print("\nBest solution chromosome found: ")
best_route_overall = new_population[best_match_idx]
for i, route_part in enumerate(best_route_overall):
    print(f"  Truck {i+1}: {route_part}")
print("Best solution fitness: ", final_fitness_scores[best_match_idx])

# --- Incinerator Trip Analysis for Best Solution ---
# This analysis reflects the user's script which may differ from fitness function's internal costing
# if the fitness function had additional implicit trip logic (which this version doesn't).
print("\n--- Incinerator Trip Analysis for Best Solution ---")
minIncineratorTrips_theoretical = math.ceil(totalTrash / truck_capacity)
print(f"Total Trash: {totalTrash}")
print(f"Truck Capacity (per truck): {truck_capacity}")
print(f"Theoretical Minimum Incinerator Trips (overall): {minIncineratorTrips_theoretical}")

evaluated_trips_in_best_solution = 0
for truck_route_chromosome_part in best_route_overall:
    current_load_sim = 0
    truck_serviced_any_bins = False
    for stop_id_analysis in truck_route_chromosome_part: # Renamed to avoid conflict
        if stop_id_analysis == incinerator_id:
            evaluated_trips_in_best_solution += 1
            current_load_sim = 0
        elif stop_id_analysis != depot_id and stop_id_analysis in bins_data:
            current_load_sim += bins_data[stop_id_analysis]['volume']
            truck_serviced_any_bins = True
    # The user's output example implies a check for mandatory final trip here for display.
    # The current fitness function in *this* script doesn't add this cost,
    # but the user's example output did have 4 trips.
    # I'll keep the display logic consistent with their example output's implication.
    if truck_serviced_any_bins and current_load_sim > 0:
        evaluated_trips_in_best_solution += 1 
print(f"Evaluated Incinerator Trips in Best Solution (explicit + display-simulated final): {evaluated_trips_in_best_solution}")

# --- Plotting Fitness ---
plt.figure(1, figsize=(10, 6)) # Explicitly use figure 1
plt.plot(best_fitness_per_gen)
plt.xlabel("Generation")
plt.ylabel("Best Fitness (Lower is Better)")
plt.title("VRP Fitness Improvement Over Generations")
plt.grid(True)

# --- Plotting Truck Routes ---
# Ensure matplotlib.pyplot is imported as plt if not already done at the top
# import matplotlib.pyplot as plt # Already imported at the top

plot_truck_routes(best_route_overall, bins_data, start_depot_location, 
                  incinerator_location, depot_id, incinerator_id, fig_num=2) # Use figure 2

plt.show() # Show all plots

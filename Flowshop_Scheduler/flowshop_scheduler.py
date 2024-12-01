import sys, os, time, random
from functools import partial
from collections import namedtuple
from itertools import product

import neighbourhood as neigh
import heuristics as heur

# Settings
TIME_LIMIT = 300.0  # Time (in seconds) to run the solver
TIME_INCREMENT = 13.0  # Time (in seconds) between heuristic measurements
DEBUG_SWITCH = False  # Displays intermediate heuristic info when True
MAX_LNS_NEIGHBOURHOODS = 1000  # Max number of neighbours to explore in LNS

def solve(data):
    """Solves an instance of the flow shop scheduling problem"""

    # Initialize strategies to avoid cyclic import issues
    initialize_strategies()
    global STRATEGIES

    # Track statistics for each strategy
    strat_improvements = {strategy: 0 for strategy in STRATEGIES}
    strat_time_spent = {strategy: 0 for strategy in STRATEGIES}
    strat_weights = {strategy: 1 for strategy in STRATEGIES}
    strat_usage = {strategy: 0 for strategy in STRATEGIES}

    # Start with a random job permutation
    perm = list(range(len(data)))
    random.shuffle(perm)

    # Track the best solution
    best_make = makespan(data, perm)
    best_perm = perm[:]
    res = best_make

    # Set up time limits and iteration tracking
    iteration = 0
    time_limit = time.time() + TIME_LIMIT
    time_last_switch = time.time()
    time_delta = TIME_LIMIT / 10
    checkpoint = time.time() + time_delta
    percent_complete = 10

    print("\nSolving...")

    while time.time() < time_limit:
        # Periodic checkpoint output
        if time.time() > checkpoint:
            print(f" {percent_complete} %")
            percent_complete += 10
            checkpoint += time_delta

        iteration += 1

        # Pick the best strategy heuristically
        strategy = pick_strategy(STRATEGIES, strat_weights)

        # Record initial state
        old_val = res
        old_time = time.time()

        # Generate candidates using the strategy's neighborhood and heuristic
        candidates = strategy.neighbourhood(data, perm)
        perm = strategy.heuristic(data, candidates)
        res = makespan(data, perm)

        # Track statistics for the chosen strategy
        strat_improvements[strategy] += res - old_val
        strat_time_spent[strategy] += time.time() - old_time
        strat_usage[strategy] += 1

        # Update best solution if improved
        if res < best_make:
            best_make = res
            best_perm = perm[:]

    # Output final statistics
    print(" 100 %\n")
    print("\nWent through %d iterations." % iteration)

    print("\n(usage) Strategy:")
    results = sorted([(strat_weights[STRATEGIES[i]], i) for i in range(len(STRATEGIES))], reverse=True)
    for (w, i) in results:
        print(f"({strat_usage[STRATEGIES[i]]}) \t{STRATEGIES[i].name}")

    return best_perm, best_make

def makespan(data, perm):
    """Computes the makespan of the provided solution"""
    return compile_solution(data, perm)[-1][-1] + data[perm[-1]][-1]

def parse_problem(file_path, instance_number=0):
    
    data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Adjust lines based on file format (skip headers, parse specific instance)
        # This example assumes each line after headers represents a job's times.
        for line in lines:
            if line.strip() and not line.startswith('#'):  # Ignore empty lines or comments
                # Parse processing times for each job on each machine
                job_times = list(map(int, line.split()))
                data.append(job_times)
                
    return data

# Define the STRATEGIES list globally
STRATEGIES = []

def initialize_strategies():
    # Example placeholder strategies
    # Replace these with actual strategies from the `heuristics` module if available
    Strategy = namedtuple('Strategy', ['name', 'neighbourhood', 'heuristic'])
    
    # Example strategies
    STRATEGIES.append(Strategy(name="Random Insertion", neighbourhood=neigh.random_insertion, heuristic=heur.random_heuristic))
    STRATEGIES.append(Strategy(name="Swap Neighborhood", neighbourhood=neigh.swap_neighbourhood, heuristic=heur.swap_heuristic))
    STRATEGIES.append(Strategy(name="Two-Opt", neighbourhood=neigh.two_opt_neighbourhood, heuristic=heur.two_opt_heuristic))

import random

def pick_strategy(strategies, strat_weights):
   
    # Create a list of weights corresponding to the order of strategies
    weights = [strat_weights[strategy] for strategy in strategies]
    
    # Use random.choices to pick a strategy based on the weights
    chosen_strategy = random.choices(strategies, weights=weights, k=1)[0]
    
    return chosen_strategy

def compile_solution(data, perm):
    
    num_jobs = len(perm)
    num_machines = len(data[0])

    # Initialize a table to store completion times
    result = [[0] * num_machines for _ in range(num_jobs)]

    # Compute the completion times for each job in each machine in the given order (perm)
    for i in range(num_jobs):
        job_index = perm[i]  # The job in the sequence
        
        for j in range(num_machines):
            if i == 0 and j == 0:
                # The first job on the first machine
                result[i][j] = data[job_index][j]
            elif i == 0:
                # First job on subsequent machines (cumulative)
                result[i][j] = result[i][j - 1] + data[job_index][j]
            elif j == 0:
                # Subsequent jobs on the first machine
                result[i][j] = result[i - 1][j] + data[job_index][j]
            else:
                # Subsequent jobs on subsequent machines (max of previous row or column)
                result[i][j] = max(result[i - 1][j], result[i][j - 1]) + data[job_index][j]

    return result

def print_solution(data, perm):
    
    # Compute the schedule based on the optimal job sequence
    schedule = compile_solution(data, perm)
    
    # Print the job sequence
    print("Optimal Job Sequence:", perm)
    
    # Print the schedule with completion times on each machine
    print("\nCompletion times on each machine:")
    for job_index, job_schedule in zip(perm, schedule):
        print(f"Job {job_index + 1}: {job_schedule}")

    # Calculate and print the makespan
    ms = makespan(data, perm)
    print("\nMinimum Makespan:", ms)

if __name__ == '__main__':

# Manually specify the path to your problem data file
    file_path = r"C:\Users\sahme\OneDrive\Desktop\systems.txt"  

    instance_number = 0  # Specify the instance number, if your file has multiple instances

# Parse the problem data
    data = parse_problem(file_path, instance_number)

# Run the solver with the parsed data
    perm, ms = solve(data)

# Print the best solution found
    print_solution(data, perm)
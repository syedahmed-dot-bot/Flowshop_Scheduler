# heuristics.py
import random

def random_heuristic(data, candidates):
    """Returns a candidate at random."""
    return random.choice(candidates)

def swap_heuristic(data, candidates):
    """Returns the first candidate as a placeholder."""
    return candidates[0]

def two_opt_heuristic(data, candidates):
    """Returns the first candidate as a placeholder."""
    return candidates[0]

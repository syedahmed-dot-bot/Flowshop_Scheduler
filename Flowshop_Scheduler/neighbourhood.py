# neighbourhood.py
import random

def random_insertion(data, perm):
    """Generates a new permutation by randomly inserting one job into a new position."""
    i, j = random.sample(range(len(perm)), 2)
    perm.insert(j, perm.pop(i))
    return [perm]

def swap_neighbourhood(data, perm):
    """Generates a new permutation by swapping two random jobs."""
    i, j = random.sample(range(len(perm)), 2)
    perm[i], perm[j] = perm[j], perm[i]
    return [perm]

def two_opt_neighbourhood(data, perm):
    """Generates a new permutation by reversing a random subsection."""
    i, j = sorted(random.sample(range(len(perm)), 2))
    perm[i:j] = perm[i:j][::-1]
    return [perm]

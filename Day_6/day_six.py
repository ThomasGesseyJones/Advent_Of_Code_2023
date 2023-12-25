"""Day Six of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
import numpy as np

# Parameters
input_file = 'input.txt'


# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Data Processing
def get_times_and_distances(input_data: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract race times and distances from the input data"""
    logging.log(logging.DEBUG, f'Extracting race times and distances')
    times = np.array([int(time) for time in input_data[0].split()[1:]])
    distances = np.array([int(distance) for distance in input_data[1].split()[1:]])
    return times, distances


# Mathematical interlude
"""
Call the time the race lasts t and the time held down for t_h.
Then the boat has velocity v=1*(t_h)  (all units mm, ms) and so travels a distance d=(t-t_h)*t_h.
We are interested in the condition where this exceeds some record r
d > r,
(t - t_h)*t_h >r,
0 > t_h^2 - t * t_h + r.
This inequality has roots
t_h = (t +- sqrt(t^2 - 4r))/2
By signature of t_h^2 term we thus know the condition to win the race is
(t - sqrt(t^2 - 4r))/2 < t_h < (t + sqrt(t^2 - 4r))/2
and so naively the range is
sqrt(t^2 - 4r).

But we are limited to integers and are left with two cases depending on if t is even or odd.
If t is even then t/2 is a whole number and the range (remembering to exclude exact ends) covers
2*ceil(sqrt(t^2 - 4r) / 2) - 1,
while if t is odd then the band is centered on a half integer and the range covers
2*ceil((sqrt(t^2 - 4r) - 1)/2) 

This should be the solution, so let us code this up.
"""

def num_ways_to_win(time_allowed: np.ndarray, distance_record: np.ndarray) -> np.ndarray:
    """Calculate the number of ways to win each race."""
    logging.log(logging.DEBUG, f'Calculating ways to win.')
    ways_to_win = np.zeros_like(time_allowed, dtype=float)  # Will convert to int later

    # Two cases
    odd_case = np.mod(time_allowed, 2).astype(bool)
    even_case = np.logical_not(odd_case)
    range_term = np.sqrt(time_allowed**2 - 4*distance_record)
    ways_to_win[even_case] = 2*np.ceil(range_term[even_case] / 2) - 1
    ways_to_win[odd_case] = 2*np.ceil((range_term[odd_case] - 1)/2)
    ways_to_win = np.round(ways_to_win)

    # If less than zero correct to zeros
    ways_to_win[ways_to_win < 0] = 0

    return ways_to_win


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input data and split into seeds and maps
    input_data = load_input()
    logging.log(logging.DEBUG, f'{input_data=}')

    # Race info
    time_allowed, distance_record = get_times_and_distances(input_data)
    logging.log(logging.DEBUG, f'{time_allowed=}')
    logging.log(logging.DEBUG, f'{distance_record=}')

    # Calculate ways to win and take their product (answer to Part 1)
    ways_to_win = num_ways_to_win(time_allowed, distance_record)
    logging.log(logging.DEBUG, f'{ways_to_win=}')
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 6 part 1, is: {np.prod(ways_to_win)}')

    # True race info
    true_time_allowed = np.array([''.join(time_allowed.astype(str))], dtype=np.uint64)
    true_distance_record = np.array([''.join(distance_record.astype(str))], dtype=np.uint64)
    logging.log(logging.DEBUG, f'{true_time_allowed=}')
    logging.log(logging.DEBUG, f'{true_distance_record=}')

    # Calculate ways to win in the one large race
    ways_to_win = num_ways_to_win(true_time_allowed, true_distance_record)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 6 part 2, is: {ways_to_win[0]}')


if __name__ == "__main__":
    main()

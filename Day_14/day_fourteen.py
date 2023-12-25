"""Day Fourteen of Advent of Code 2023."""

# Import required packages
import logging
import numpy as np
from enum import Enum


# Parameters
input_file = 'input.txt'


# Constants
round_rock_symbol = 'O'
square_rock_symbol = '#'
empty_space_symbol = '.'
total_cycles = 1000000000  # For Part II


# IO
def load_rock_map() -> np.ndarray:
    """Load the rock map from the input file."""
    logging.log(logging.DEBUG, f'Loading rock map from {input_file}.')
    with open(input_file, 'r') as f:
        rock_map = np.array([[symbol for symbol in line.strip()] for line in f.readlines()])
    return rock_map


# Geometry
class Direction(Enum):
    """An enum for the directions."""
    north = 0
    east = 1
    south = 2
    west = 3


def tilt_map(rock_map: np.ndarray, direction: Direction) -> np.ndarray:
    """Tilt the map in the given direction."""
    # Rotate the map so always tilting north
    rock_map = rotate_to_north(rock_map, direction)

    # Consider each column in turn and tilt it
    for col_idx in range(rock_map.shape[1]):
        # Get the old column information
        old_col = rock_map[:, col_idx]

        # Find the position of all square rocks, prepend -1 to represent the top of the column, and the length of the
        # column to represent the bottom of the column
        square_rock_indices = np.argwhere(old_col == square_rock_symbol)
        square_rock_indices = np.insert(square_rock_indices, 0, -1)
        square_rock_indices = np.append(square_rock_indices, len(old_col))

        # Consider rocks falling in the inter-square rock gap
        new_col = []
        for lower_sq_rock_idx, higher_sq_rock_idx in zip(square_rock_indices[:-1], square_rock_indices[1:]):
            # Add the square rock to the new column
            new_col.append(square_rock_symbol)

            # Then add the number of round rocks that would fall in the gap
            num_round_rocks = np.sum(old_col[lower_sq_rock_idx + 1:higher_sq_rock_idx] == round_rock_symbol)
            new_col = new_col + [round_rock_symbol] * num_round_rocks

            # Then add the empty space
            num_empty_spaces = higher_sq_rock_idx - lower_sq_rock_idx - num_round_rocks - 1
            new_col = new_col + [empty_space_symbol] * num_empty_spaces

        # Replace the old column with the new column
        rock_map[:, col_idx] = new_col[1:]  # Remove the first square rock, which was added to the new column


    # Rotate the map back to its original orientation
    if direction == Direction.north:
        pass
    elif direction == Direction.east:
        rock_map = np.rot90(rock_map, k=3)
    elif direction == Direction.south:
        rock_map = np.rot90(rock_map, k=2)
    elif direction == Direction.west:
        rock_map = np.rot90(rock_map, k=1)

    return rock_map


def rotate_to_north(rock_map: np.ndarray, direction: Direction) -> np.ndarray:
    """Rotate the map so the given direction is now north."""
    if direction == Direction.north:
        logging.log(logging.DEBUG, f'Tilting map north.')
    elif direction == Direction.east:
        logging.log(logging.DEBUG, f'Tilting map east.')
        rock_map = np.rot90(rock_map, k=1)
    elif direction == Direction.south:
        logging.log(logging.DEBUG, f'Tilting map south.')
        rock_map = np.rot90(rock_map, k=2)
    elif direction == Direction.west:
        logging.log(logging.DEBUG, f'Tilting map west.')
        rock_map = np.rot90(rock_map, k=3)
    return rock_map


def rock_cycle(rock_map: np.ndarray) -> np.ndarray:
    """Perform one rock cycle."""
    # Tilt the map in each direction
    rock_map = tilt_map(rock_map, Direction.north)
    rock_map = tilt_map(rock_map, Direction.west)
    rock_map = tilt_map(rock_map, Direction.south)
    rock_map = tilt_map(rock_map, Direction.east)
    return rock_map


def perform_rock_cycles(rock_map: np.ndarray, num_cycles: int) -> np.ndarray:
    """Perform the given number of rock cycles."""
    # Each cycle operation only depends on the current rock map state (Markovian), and the map is finite, so we will
    # eventually reach a state we have seen before and enter a loop. In fact, given the huge number of cycles, we are
    # asked to do this will likely happen very quickly. Once we are in the loop we can avoid doing any more cycles by
    # analytically calculating which state in the loop we will be in after the given number of cycles.

    # Keep track of past maps to detect the loop (idx in past_maps is the cycle number)
    past_maps = [rock_map.copy()]
    for _ in range(num_cycles):  # Worst case scenario, we do all the cycles
        rock_map = rock_cycle(rock_map)

        # Check if we've seen this map before
        if np.any([np.array_equal(past_map, rock_map) for past_map in past_maps]):
            break

        # If not, add it to the list of past maps
        past_maps.append(rock_map.copy())
    else:
        # We did it the hard way
        return rock_map

    # Loop was detected so find the index of the first repeated map and the length of the loop
    first_instance_idx = np.argwhere([np.array_equal(past_map, rock_map) for past_map in past_maps])[0][0]
    loop_length = len(past_maps) - first_instance_idx

    # Use loop information to find the map at the end of the cycles
    loops_total = (num_cycles - first_instance_idx) // loop_length
    idx_of_end = num_cycles - loops_total * loop_length
    rock_map = past_maps[idx_of_end]
    return rock_map


def count_load(rock_map: np.ndarray, direction: Direction) -> int:
    """Count the load in the rock map against that direction edge."""
    # Rotate the map so always tilting north
    rock_map = rotate_to_north(rock_map, direction)

    # Find all the round rocks in the map
    round_rock_indices = np.argwhere(rock_map == round_rock_symbol)

    # Calculate the load of each round rock
    map_size = rock_map.shape[0]
    rock_loads = [map_size - rock_idx for rock_idx in round_rock_indices[:, 0]]

    return sum(rock_loads)


def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    rock_map = load_rock_map()
    logging.log(logging.DEBUG, f'Loaded map')

    # Part I, tilt north and count the load
    rock_map = tilt_map(rock_map, Direction.north)
    load = count_load(rock_map, Direction.north)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 14 part 1, is: {load}')

    # Part II, perform the cycles and count the load
    rock_map = load_rock_map()  # Reload the map
    rock_map = perform_rock_cycles(rock_map, total_cycles)
    load = count_load(rock_map, Direction.north)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 14 part 2, is: {load}')


if __name__ == '__main__':
    main()


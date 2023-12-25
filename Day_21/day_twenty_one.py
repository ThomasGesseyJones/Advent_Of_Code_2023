"""Day Twenty One of Advent of Code 2023."""

# Import required packages
import numpy as np
import logging
from typing import Tuple
from bisect import insort


# Parameters
input_file = 'input.txt'


# Constants
symbol_start = 'S'
symbol_garden = '.'
symbol_rock = '#'
part_1_steps = 64
part_2_steps = 26501365


# IO
def load_input(file_path: str) -> Tuple[np.ndarray, int, int]:
    """Load the input file, and extract the garden map and starting position."""
    logging.log(logging.DEBUG, f'Loading input file: {file_path}')

    with open(file_path, 'r') as f:
        # Read the file line by line, and store the garden map as a list of lists of booleans
        # True = garden, False = rock
        garden_map = []
        for row, line in enumerate(f):
            line = line.strip()
            line_list = []
            for col, symbol in enumerate(line):
                if symbol == symbol_start:
                    # Start position is a garden, but we need to know where it is
                    start_row = row
                    start_col = col
                    line_list.append(True)
                elif symbol == symbol_garden:
                    line_list.append(True)
                elif symbol == symbol_rock:
                    line_list.append(False)
                else:
                    raise ValueError(f'Unknown symbol: {symbol}')
            garden_map.append(line_list)

        # Convert the garden map to a numpy array for easier indexing
        garden_map = np.array(garden_map)
        return garden_map, start_row, start_col


# Data processing
def calculate_minimum_steps_to_reach(garden_map: np.ndarray, start_row: int, start_col: int) -> np.ndarray:
    """Calculate the minimum number of steps to reach each tile in the garden map.

    This is Dijkstra's Algorithm on a grid.
    """
    logging.log(logging.DEBUG, f'Calculating minimum steps to reach each tile in the garden map')

    # Data structures to store the minimum number of steps to reach each tile
    minimum_steps_to_reach = np.full(garden_map.shape, np.inf)
    minimum_steps_to_reach[start_row, start_col] = 0

    # Useful variables
    min_row = 0
    max_row = garden_map.shape[0] - 1
    min_col = 0
    max_col = garden_map.shape[1] - 1

    # Active node list
    active_nodes = [(start_row, start_col)]

    # Dijkstra's Algorithm
    while len(active_nodes) > 0:
        # Get the current node
        current_node = active_nodes.pop(0)
        current_distance = minimum_steps_to_reach[current_node]

        # Get the current node's neighbors
        current_row, current_col = current_node
        potential_neighbors = [(current_row - 1, current_col),
                               (current_row + 1, current_col),
                               (current_row, current_col - 1),
                               (current_row, current_col + 1)]

        # Check each neighbor
        for potential_neighbor in potential_neighbors.copy():
            potential_row, potential_col = potential_neighbor

            # Skip any that are out of bounds, rocks, or we already have a better path to
            # Out of bounds
            if potential_row < min_row or potential_row > max_row or \
               potential_col < min_col or potential_col > max_col:
                continue

            # Rock
            if not garden_map[potential_row, potential_col]:
                continue

            # Already have a better path
            if minimum_steps_to_reach[potential_row, potential_col] <= current_distance + 1:
                continue

            # Update the minimum steps to reach and place at the appropriate position in active nodes
            minimum_steps_to_reach[potential_row, potential_col] = current_distance + 1
            if (potential_row, potential_col) in active_nodes:
                active_nodes.remove((potential_row, potential_col))
            insort(active_nodes, (potential_row, potential_col))

    return minimum_steps_to_reach


def can_reach_in_n_steps(minimum_steps_to_reach: np.ndarray, n: int) -> np.ndarray:
    """Calculate the number of tiles that can be reached in exactly n steps"""
    # A tile can be reached in exactly n steps if the minimum steps to reach it is less than or equal to n and the
    # same parity as n
    can_be_reached = np.logical_and(minimum_steps_to_reach <= n, minimum_steps_to_reach % 2 == n % 2)
    return can_be_reached


def can_reach_in_large_n_steps(garden_map: np.ndarray,
                               start_row: int,
                               start_col: int,
                               n: int) -> int:
    """Calculate the number of tiles that can be reached in exactly n steps, where n is large.

    Now with infinite tiling of the garden map!

    Note that there are 'paths' of garden tiles from S to the edge of the garden map, in each of the four directions
    and around the edge of the garden map. Hence, the edges of the garden map can all be reached in the minimum
    number of steps, the 1-norm distance from S to that point. As a result, we can very quickly calculate the number
    of garden copies that can be reached in a number of steps and how many it takes to get there.
    They will be grouped into a small number of groups, depending on how many steps it takes to reach the edge of that
    garden copy. Hence, allowing for the basis of an efficient calculation.
    """
    logging.log(logging.DEBUG, f'Calculating the number of tiles that can be reached in exactly {n} steps')

    # Useful variables
    n_parity = n % 2
    map_rows, map_cols = garden_map.shape
    assert map_rows == map_cols  # We assume the garden map is square
    assert map_rows % 2 == 1  # We assume the garden map has an odd number of rows and columns
    garden_size = map_rows

    straight_directions = ['north', 'south', 'west', 'east']
    diagonal_directions = ['north_west', 'north_east', 'south_west', 'south_east']

    # Compute the minimum number of steps to reach each tile in the central garden map
    minimum_steps_to_reach_center = calculate_minimum_steps_to_reach(garden_map, start_row, start_col)

    # And equivalent for entering from the eight cardinal directions
    enter_from_north = calculate_minimum_steps_to_reach(garden_map, 0, garden_size // 2 )
    enter_from_south = calculate_minimum_steps_to_reach(garden_map, garden_size - 1, garden_size // 2)
    enter_from_west = calculate_minimum_steps_to_reach(garden_map, garden_size // 2, 0)
    enter_from_east = calculate_minimum_steps_to_reach(garden_map, garden_size // 2, garden_size - 1)
    enter_from_north_west = calculate_minimum_steps_to_reach(garden_map, 0, 0)
    enter_from_north_east = calculate_minimum_steps_to_reach(garden_map, 0, garden_size - 1)
    enter_from_south_west = calculate_minimum_steps_to_reach(garden_map, garden_size - 1, 0)
    enter_from_south_east = calculate_minimum_steps_to_reach(garden_map, garden_size - 1, garden_size - 1)
    minimum_steps_to_reach = {
            'north': enter_from_north,
            'south': enter_from_south,
            'west': enter_from_west,
            'east': enter_from_east,
            'north_west': enter_from_north_west,
            'north_east': enter_from_north_east,
            'south_west': enter_from_south_west,
            'south_east': enter_from_south_east}

    # Add the middles' contribution to the number of tiles that can be reached in n steps
    can_be_reached = can_reach_in_n_steps(minimum_steps_to_reach_center, n)
    num_can_be_reached = np.sum(can_be_reached)

    # Loop over garden copies that can be reached adding their contribution to the number of tiles that can be reached
    # in n steps
    minimum_to_reach_edge = garden_size // 2 + 1
    num_on_diagonals = 0

    # Start with any garden maps that can be fully reached.
    # Find the furthest distance from S in current ring (they are actually diagonal diamonds)
    furthest_distance = {k: int(np.max(v[v != np.inf])) for k, v in minimum_steps_to_reach.items()}
    for direction, distance in furthest_distance.items():
        if direction in straight_directions:
            furthest_distance[direction] += minimum_to_reach_edge
        elif direction in diagonal_directions:
            furthest_distance[direction] += 2*minimum_to_reach_edge

    # Record how many full maps of each parity to include in the final result
    # First is even, second is odd
    copies_to_include = {k: [0, 0] for k in straight_directions + diagonal_directions}
    while np.all([v <= n for v in furthest_distance.values()]):
        # 1 more on each diagonal each time
        num_on_diagonals += 1

        # Add the number of full maps of each parity to include in the final result
        straight_direction_parity = minimum_to_reach_edge % 2
        diagonal_direction_parity = (minimum_to_reach_edge + garden_size // 2 + 1) % 2
        for direction in straight_directions:
            copies_to_include[direction][straight_direction_parity] += 1
        for direction in diagonal_directions:
            copies_to_include[direction][diagonal_direction_parity] += num_on_diagonals

        # Update loop variables
        furthest_distance = {k: v + garden_size for k, v in furthest_distance.items()}
        minimum_to_reach_edge += garden_size

    # Add the contribution of each full map to the number of tiles that can be reached in n steps
    for direction in straight_directions + diagonal_directions:
        num_even, num_odd = copies_to_include[direction]
        minimum_steps = minimum_steps_to_reach[direction]
        num_even_minimum_steps = np.sum(np.logical_and(minimum_steps < np.inf, minimum_steps % 2 == 0))
        num_odd_minimum_steps = np.sum(np.logical_and(minimum_steps < np.inf, minimum_steps % 2 == 1))
        num_minimum_steps = [num_even_minimum_steps, num_odd_minimum_steps]
        num_can_be_reached += num_even*num_minimum_steps[n_parity] + num_odd*num_minimum_steps[1 - n_parity]


    # Deal with the outer edge, where we can't reach the whole of the garden map and so need to be more careful
    while minimum_to_reach_edge <= n:
        # Vertically or horizontally aligned with the central garden map
        for direction in ['north', 'south', 'west', 'east']:
            minimum_to_reach_map = minimum_steps_to_reach[direction] + minimum_to_reach_edge
            can_be_reached = can_reach_in_n_steps(minimum_to_reach_map, n)
            num_can_be_reached += np.sum(can_be_reached)

        # Diagonals
        num_on_diagonals += 1
        for direction in ['north_west', 'north_east', 'south_west', 'south_east']:
            minimum_to_reach_map = minimum_steps_to_reach[direction] + minimum_to_reach_edge + garden_size // 2 + 1
            can_be_reached = can_reach_in_n_steps(minimum_to_reach_map, n)
            num_can_be_reached += np.sum(can_be_reached)*num_on_diagonals

        minimum_to_reach_edge += garden_size

    return num_can_be_reached


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    garden_map, start_row, start_col = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded garden map with shape: {garden_map.shape}')

    # Part I, find the number of tiles that can be reached in exactly 64 steps
    minimum_steps_to_reach = calculate_minimum_steps_to_reach(garden_map, start_row, start_col)
    can_reach_at_64 = can_reach_in_n_steps(minimum_steps_to_reach, part_1_steps)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 21 part 1, is: '
                                f'{np.sum(can_reach_at_64)}')

    # Part II, find the number of tiles that can be reached in exactly 26501365 steps
    # This is far too many to calculate using Dijkstra's Algorithm, so we need to find a pattern.
    can_reach_at_big_number = can_reach_in_large_n_steps(garden_map, start_row, start_col, part_2_steps)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 21 part 2, is: '
                                f'{can_reach_at_big_number}')


if __name__ == '__main__':
    main()

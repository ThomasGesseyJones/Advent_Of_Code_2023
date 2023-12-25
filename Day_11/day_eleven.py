"""Day Eleven of Advent of Code 2023."""

# Import required packages
import logging
import numpy as np

# Parameters
input_file = 'input.txt'

# Constants
symbol_empty_space = '.'
code_empty_space = 0
symbol_galaxy = '#'
code_galaxy = 1
conversion_dict = {symbol_empty_space: code_empty_space, symbol_galaxy: code_galaxy}

expansion_factor = 1_000_000  # For part 2


# IO
def load_galaxy_map() -> np.ndarray:
    """Load the galaxy map from the input file."""
    with open(input_file, 'r') as f:
        symbol_map = [[symbol for symbol in line.strip()] for line in f.readlines()]
        code_map = [[conversion_dict[symbol] for symbol in line] for line in symbol_map]
        return np.array(code_map)


# 'Physics'
def account_for_expansion(galaxy_map: np.ndarray) -> np.ndarray:
    """Account for the expansion of the universe."""
    logging.log(logging.DEBUG, f'Accounting for expansion of the universe.')

    # Find columns of all empty space
    empty_space_columns = np.where(np.all(galaxy_map == code_empty_space, axis=0))[0]

    # Add an extra empty column to the right of the empty space columns
    new_galaxy_map = np.insert(galaxy_map, empty_space_columns + 1, code_empty_space, axis=1)

    # Do the same for the rows
    empty_space_rows = np.where(np.all(new_galaxy_map == code_empty_space, axis=1))[0]
    new_galaxy_map = np.insert(new_galaxy_map, empty_space_rows + 1, code_empty_space, axis=0)

    return new_galaxy_map


# Geometry
def calculate_min_distances(galaxy_x_pos: np.ndarray, galaxy_y_pos: np.ndarray) -> np.ndarray:
    """Calculate the one norm minimum distances between galaxies."""
    logging.log(logging.DEBUG, f'Calculating minimum distances between galaxies.')

    # One norm distance = abs(galaxy_1_x - galaxy_2_x) + abs(galaxy_1_y - galaxy_2_y)
    # This can be done very nicely with square matrices
    galaxy_1_x_matrix = np.tile(galaxy_x_pos, (len(galaxy_x_pos), 1))
    galaxy_2_x_matrix = np.transpose(galaxy_1_x_matrix)  # Galaxy 1 is the row, galaxy 2 is the column
    galaxy_1_y_matrix = np.tile(galaxy_y_pos, (len(galaxy_y_pos), 1))
    galaxy_2_y_matrix = np.transpose(galaxy_1_y_matrix)  # Galaxy 1 is the row, galaxy 2 is the column
    distance = np.abs(galaxy_1_x_matrix - galaxy_2_x_matrix) + np.abs(galaxy_1_y_matrix - galaxy_2_y_matrix)

    # Remove the duplicates above the diagonal
    distance = np.tril(distance)
    return distance


def correct_distances_for_expansion(base_distances: np.ndarray, base_galaxy_map: np.ndarray,
                                    galaxy_x_pos: np.ndarray, galaxy_y_pos: np.ndarray) -> np.ndarray:
    """Correct a set of intergalactic distances for the expansion of the universe."""
    logging.log(logging.DEBUG, f'Correcting distances for expansion of the universe.')

    # Avoid side effects
    distances = np.copy(base_distances)

    # Find the columns and rows of all empty space
    empty_space_columns = np.where(np.all(base_galaxy_map == code_empty_space, axis=0))[0]
    empty_space_rows = np.where(np.all(base_galaxy_map == code_empty_space, axis=1))[0]

    # Check
    assert all([not empty_col in galaxy_x_pos for empty_col in empty_space_columns])
    assert all([not empty_row in galaxy_y_pos for empty_row in empty_space_rows])

    # Loop over all distances (e.g. a lower diagonal matrix) correcting if needed
    for galaxy_1_idx, (galaxy_1_x, galaxy_1_y) in enumerate(zip(galaxy_x_pos, galaxy_y_pos)):
        for galaxy_2_idx, (galaxy_2_x, galaxy_2_y) in (
                enumerate(zip(galaxy_x_pos[:galaxy_1_idx], galaxy_y_pos[:galaxy_1_idx]))):
            # Find range of space between the two galaxies
            inter_x_range = range(min(galaxy_1_x, galaxy_2_x), max(galaxy_1_x, galaxy_2_x) + 1)
            inter_y_range = range(min(galaxy_1_y, galaxy_2_y), max(galaxy_1_y, galaxy_2_y) + 1)

            # Find number of empty columns and rows between the two galaxies
            num_empty_x = sum([empty_col in inter_x_range for empty_col in empty_space_columns])
            num_empty_y = sum([empty_row in inter_y_range for empty_row in empty_space_rows])

            # Correct the distance as needed
            correct_factor = (expansion_factor - 1) * (num_empty_x + num_empty_y)
            distances[galaxy_1_idx, galaxy_2_idx] += correct_factor

    return distances


def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    base_galaxy_map = load_galaxy_map()
    logging.log(logging.DEBUG, f'{base_galaxy_map.shape[0]} rows and {base_galaxy_map.shape[1]} columns.')

    # Correct map for expansion of the universe
    corrected_galaxy_map = account_for_expansion(base_galaxy_map)
    logging.log(logging.DEBUG, f'{corrected_galaxy_map.shape[0]} rows and '
                               f'{corrected_galaxy_map.shape[1]} columns.')

    # Find minimum distances between galaxies, the sum of which is the answer to the first part
    galaxy_x_pos, galaxy_y_pos = np.where(corrected_galaxy_map == code_galaxy)
    minimum_inter_galactic_distances = calculate_min_distances(galaxy_x_pos, galaxy_y_pos)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 11 part 1, is: '
                              f'{np.sum(minimum_inter_galactic_distances)}')

    # With the million fold expansion it will be much quicker to correct at the distance calculation stage
    # for the expansion of the universe rather than padding the map
    galaxy_y_pos, galaxy_x_pos = np.where(base_galaxy_map == code_galaxy)
    base_distances = calculate_min_distances(galaxy_x_pos, galaxy_y_pos)
    corrected_distances = correct_distances_for_expansion(base_distances, base_galaxy_map, galaxy_x_pos, galaxy_y_pos)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 11 part 2, is: '
                              f'{np.sum(corrected_distances)}')


if __name__ == '__main__':
    main()

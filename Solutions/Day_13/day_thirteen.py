"""Day Thirteen of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
import numpy as np
import tqdm

# Parameters
input_file = 'input.txt'

# Constants
rocks_symbol = '#'
ash_symbol = '.'
rocks_code = 1
ash_code = 0
symbols_to_codes = {rocks_symbol: rocks_code, ash_symbol: ash_code}

vertical_weighting = 1
horizontal_weighting = 100

# IO
def load_rock_map() -> List[np.ndarray]:
    """Load the rock map from the input file."""
    logging.log(logging.DEBUG, f'Loading rock maps from {input_file}.')
    with open(input_file, 'r') as f:
        # Store the maps in a list
        rock_maps = []
        current_map_so_far = []

        # Build up maps from the input file
        for line in f.readlines():
            # Blank line means the end of the current map and start of a new map
            if line.strip() == '':
                rock_maps.append(np.array(current_map_so_far))
                current_map_so_far = []
                continue

            # Else need to add the line to the current map
            line_symbols = [symbol for symbol in line.strip()]
            line_codes = [symbols_to_codes[symbol] for symbol in line_symbols]
            current_map_so_far.append(line_codes)

        # Add the last map
        rock_maps.append(np.array(current_map_so_far))

    return rock_maps


# Data processing
def find_mirror_lines_in_map(rock_map: np.ndarray, allowed_num_smudges: Tuple = (0,)) -> Tuple[List[int], List[int]]:
    """Find the mirror lines in a map.

    Returns two lists. The positions of any vertical mirror lines in the first list, and the positions of any horizontal
    mirror lines in the second list. Numbering such that a vertical mirror at position x has x columns to its left, and
    a horizontal mirror at position y has y rows above it.

    allowed_num_smudges is the number(s) of smudges we expect to find in the map. E.g., the number of cells that deviate
    from exact symmetry. We will only return mirror lines with these numbers of smudges.
    """
    # Start with the vertical mirror lines
    vertical_mirror_lines = find_vertical_mirror_lines_in_map(rock_map, allowed_num_smudges)

    # Can find horizontal mirror lines by transposing the map and then finding the new arrays vertical mirror lines
    horizontal_mirror_lines = find_vertical_mirror_lines_in_map(np.transpose(rock_map), allowed_num_smudges)

    return vertical_mirror_lines, horizontal_mirror_lines


def find_vertical_mirror_lines_in_map(rock_map: np.ndarray, allowed_num_smudges: Tuple = (0,)) -> List[int]:
    """Find the vertical mirror lines in a map.
    
    Returns a list of the positions of any vertical mirror lines in the map. Numbering such that a vertical mirror at
    position x has x columns to its left.

    allowed_num_smudges is the number(s) of smudges we expect to find in the map. E.g., the number of cells that deviate
    from exact vertical symmetry. We will only return mirror lines with these numbers of smudges.
    """
    vertical_mirror_lines = []

    # Brute force search over possible positions as not that many at most 9,900 (99 * 100)
    for mirror_position in range(1, rock_map.shape[1]):
        # Find the columns we need to compare for mirroring, right_cols are flipped so we can compare them
        # directly to the left_cols
        left_cols, right_cols = columns_to_compare_for_mirroring(mirror_position, rock_map)

        # If mirrored along the mirror_position the left and right columns should now be the same
        # For n smudges, the number of different elements will be n
        if np.sum(left_cols != right_cols) in allowed_num_smudges:
            vertical_mirror_lines.append(mirror_position)

    return vertical_mirror_lines


def columns_to_compare_for_mirroring(mirror_position: int, rock_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find the columns that need to be compared to determine if mirroring about at a given position."""
    # Find how many columns we need to compare
    columns_to_compare = min(mirror_position, rock_map.shape[1] - mirror_position)

    # Isolate those columns
    left_cols = rock_map[:, mirror_position - columns_to_compare:mirror_position]
    right_cols = rock_map[:, mirror_position:mirror_position + columns_to_compare]

    # Flip the right columns
    right_cols = np.flip(right_cols, axis=1)
    return left_cols, right_cols


def find_all_mirror_lines(rock_maps: List[np.ndarray],
                          allowed_num_smudges: Tuple = (0,)) -> List[Tuple[List[int], List[int]]]:
    """Find all the mirror lines in all the maps."""
    logging.log(logging.DEBUG, f'Finding all mirror lines in all maps.')

    # Loop over all maps finding the mirror lines in each
    map_idx = 0
    all_mirror_lines = []
    for rock_map in tqdm.tqdm(rock_maps):  # Progress bar!
        map_idx += 1
        vertical_mirror_lines, horizontal_mirror_lines = find_mirror_lines_in_map(rock_map, allowed_num_smudges)
        all_mirror_lines.append((vertical_mirror_lines, horizontal_mirror_lines))

    return all_mirror_lines


def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    rock_maps = load_rock_map()
    logging.log(logging.DEBUG, f'Loaded {len(rock_maps)} maps')

    # Part I, find the lines of mirroring in each map.
    # Part II, correct the smudges and repeat
    for part_num in range(1, 3):
        # Turn on smudge detection if part II
        if part_num == 2:
            all_mirror_lines = find_all_mirror_lines(rock_maps, (1,))
        else:
            all_mirror_lines = find_all_mirror_lines(rock_maps)

        mirror_scores = [np.sum(vert_mirrors)*vertical_weighting + \
                         np.sum(horiz_mirrors)*horizontal_weighting for vert_mirrors, horiz_mirrors in all_mirror_lines]
        mirror_scores = [int(score) for score in mirror_scores]
        logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 13 part {part_num}, is: '
                                  f'{sum(mirror_scores)}')


if __name__ == '__main__':
    main()

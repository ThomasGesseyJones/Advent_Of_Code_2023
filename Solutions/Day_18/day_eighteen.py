"""Day Eighteen of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
import numpy as np
from enum import Enum, auto


# Parameters
input_file = 'input.txt'


# Data structures
class Direction(Enum):
    """An enum for the directions."""
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()


symbol_to_direction = {
    'U': Direction.UP,
    'R': Direction.RIGHT,
    'D': Direction.DOWN,
    'L': Direction.LEFT
}


number_to_direction = {
    0: Direction.RIGHT,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.UP
}



# IO
def load_dig_plan(file_path: str) -> List[Tuple[Direction, int, str]]:
    """Load the dig plan from the input file."""
    with open(file_path, 'r') as f:
        dig_plan = [line.strip() for line in f.readlines()]
        dig_plan = [line.split() for line in dig_plan]
        dig_plan = [(symbol_to_direction[line[0]], int(line[1]), line[2]) for line in dig_plan]
    return dig_plan


# Digging
def dig_hole(dig_plan: List[Tuple[Direction, int]]) -> int:
    """Dig the hole according to the dig plan. Return the volume of the hole."""
    logging.log(logging.DEBUG, f'Digging hole based on dig plan')

    # Make a list of edge nodes
    boundary_coords = [[0, 0]]
    for direction, length in dig_plan:
        if direction == Direction.UP:
            boundary_coords.append([boundary_coords[-1][0], boundary_coords[-1][1] + length])
        elif direction == Direction.RIGHT:
            boundary_coords.append([boundary_coords[-1][0] + length, boundary_coords[-1][1]])
        elif direction == Direction.DOWN:
            boundary_coords.append([boundary_coords[-1][0], boundary_coords[-1][1] - length])
        elif direction == Direction.LEFT:
            boundary_coords.append([boundary_coords[-1][0] - length, boundary_coords[-1][1]])

    # Remove the last duplicate node
    assert boundary_coords[-1][0] == 0 and boundary_coords[-1][1] == 0
    boundary_coords = boundary_coords[:-1]

    # Loop through segments of the boundary contracting them if possible and adding the area to the total
    total_area = 0
    segment_index = 0
    while len(boundary_coords) > 4:
        # Get the segment nodes and the leg nodes its connected to
        segment_node_1 = boundary_coords[segment_index]
        segment_node_2 = boundary_coords[(segment_index + 1) % len(boundary_coords)]
        leg_node_1 = boundary_coords[(segment_index - 1) % len(boundary_coords)]
        leg_node_2 = boundary_coords[(segment_index + 2) % len(boundary_coords)]

        # Determine the orientation of the segment
        vertical = segment_node_1[0] == segment_node_2[0]  # horizontal if false

        # Determine if the segment can be contracted
        # First setup the indices we will use
        if vertical:
            fixed_idx = 0
            varying_idx = 1
        else:
            fixed_idx = 1
            varying_idx = 0

        # Determine lengths of the segments legs
        leg_1_length = leg_node_1[fixed_idx] - segment_node_1[fixed_idx]
        leg_2_length = leg_node_2[fixed_idx] - segment_node_2[fixed_idx]

        # Legs need to be in the same direction to contract
        if leg_1_length * leg_2_length < 0:
            segment_index = (segment_index + 1) % len(boundary_coords)
            continue

        # Determine which length we would contract (the shorter one)
        if abs(leg_1_length) < abs(leg_2_length):
            length_to_contract = leg_1_length
            contracted_num = 1
        elif abs(leg_1_length) > abs(leg_2_length):
            length_to_contract = leg_2_length
            contracted_num = 2
        else:
            length_to_contract = leg_1_length
            contracted_num = 0

        # Determine if we could contract the leg (no nodes in the way)
        # First find the nodes that we would contract to
        new_segment_node_1 = segment_node_1.copy()
        new_segment_node_2 = segment_node_2.copy()
        new_segment_node_1[fixed_idx] += length_to_contract
        new_segment_node_2[fixed_idx] += length_to_contract

        # Contract box dimensions
        min_x = np.min([new_segment_node_1[0], new_segment_node_2[0], segment_node_1[0], segment_node_2[0]])
        max_x = np.max([new_segment_node_1[0], new_segment_node_2[0], segment_node_1[0], segment_node_2[0]])
        min_y = np.min([new_segment_node_1[1], new_segment_node_2[1], segment_node_1[1], segment_node_2[1]])
        max_y = np.max([new_segment_node_1[1], new_segment_node_2[1], segment_node_1[1], segment_node_2[1]])

        # No node except the segment or leg nodes can be in the box
        can_contract = True
        for idx, node in enumerate(boundary_coords):
            # These nodes are allowed to be in the box
            if idx == segment_index or idx == (segment_index + 1) % len(boundary_coords) or \
               idx == (segment_index - 1) % len(boundary_coords) or \
               idx == (segment_index + 2) % len(boundary_coords):
                continue

            # Check if the node is in the box
            if min_x <= node[0] <= max_x and min_y <= node[1] <= max_y:
                can_contract = False
                break

        # If we can't contract the segment, move on to the next one
        if not can_contract:
            segment_index = (segment_index + 1) % len(boundary_coords)
            continue

        # Check we are contracting, e.g. the direction of motion is inward
        phantom_point = segment_node_1.copy()
        phantom_point[varying_idx] += 0.1*(segment_node_2[varying_idx] - segment_node_1[varying_idx]) / \
                                          abs(segment_node_2[varying_idx] - segment_node_1[varying_idx])
        number_intersections_on_way_out = 0
        for idx, node in enumerate(boundary_coords):
            # Ignore own segment
            if idx == segment_index:
                continue

            # Check if will cross segment on way out
            test_segment_node_1 = node
            test_segment_node_2 = boundary_coords[(idx + 1) % len(boundary_coords)]
            test_orientation = test_segment_node_1[0] == test_segment_node_2[0]

            # Can only intersect if orientation is the same
            if test_orientation != vertical:
                continue

            # Check if on the correct side
            if not (test_segment_node_1[fixed_idx] - phantom_point[fixed_idx]) / length_to_contract  > 0:
                continue

            # And will intersect
            test_segment_upper_coord = max(test_segment_node_1[varying_idx], test_segment_node_2[varying_idx])
            test_segment_lower_coord = min(test_segment_node_1[varying_idx], test_segment_node_2[varying_idx])
            if test_segment_lower_coord <= phantom_point[varying_idx] <= test_segment_upper_coord:
                number_intersections_on_way_out += 1

        # If we will cross an odd number of segments on the way out, we are contracting. Even number, we are expanding
        if number_intersections_on_way_out % 2 == 0:
            segment_index = (segment_index + 1) % len(boundary_coords)
            continue

        # Otherwise we can contract the segment
        area_trimmed = (abs(length_to_contract)) * (abs(segment_node_1[varying_idx] - segment_node_2[varying_idx]) + 1)
        logging.log(logging.DEBUG, f'Adding {area_trimmed} to the total area')
        logging.log(logging.DEBUG, f'Contracting segments {segment_node_1} and {segment_node_2} to '
                                   f'{new_segment_node_1} and {new_segment_node_2}')
        total_area += area_trimmed
        boundary_coords[segment_index] = new_segment_node_1
        boundary_coords[(segment_index + 1)  % len(boundary_coords)] = new_segment_node_2

        # Remove the unnecessary duplicate nodes
        if contracted_num == 1:
            remove_1 = True
            remove_2 = False
        elif contracted_num == 2:
            remove_1 = False
            remove_2 = True
        else:
            remove_1 = True
            remove_2 = True

        if remove_1:
            boundary_coords = [node for node in boundary_coords if node[0] != new_segment_node_1[0] or
                               node[1] != new_segment_node_1[1]]
        if remove_2:
            boundary_coords = [node for node in boundary_coords if node[0] != new_segment_node_2[0] or
                               node[1] != new_segment_node_2[1]]

        logging.log(logging.DEBUG, f'{len(boundary_coords)} nodes remaining')
        logging.log(logging.DEBUG, f'############################################')

        # Update the segment index
        segment_index = segment_index % len(boundary_coords)

    # Add the area of the final rectangle
    top_coord = max(boundary_coords, key=lambda x: x[1])[1]
    bottom_coord = min(boundary_coords, key=lambda x: x[1])[1]
    left_coord = min(boundary_coords, key=lambda x: x[0])[0]
    right_coord = max(boundary_coords, key=lambda x: x[0])[0]
    total_area += (top_coord - bottom_coord + 1) * (right_coord - left_coord + 1)

    return total_area


def unpack_dig_plan(dig_plan: List[Tuple[Direction, int, str]]) -> List[Tuple[Direction, int]]:
    """Unpack the dig plan to a list of directions and lengths."""
    logging.log(logging.DEBUG, f'Unpacking dig plan to true dig plan')

    # Unpack the hex codes
    true_dig_plan = []
    for _, _, hex_code in dig_plan:
        # Remove superfluous characters
        hex_code = hex_code.replace('(', '').replace(')', '').replace('#', '')

        # Split the hex code into the first five digits and the last digit
        first_five_digit = hex_code[:5]
        last_digit = int(hex_code[-1])

        # Convert the first five digits to decimal
        length = int(first_five_digit, 16)

        # Convert the last digit to a direction
        direction = number_to_direction[last_digit]

        true_dig_plan.append((direction, length))

    return true_dig_plan


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    dig_plan = load_dig_plan(input_file)
    logging.log(logging.DEBUG, f'Loaded dig plan with length {len(dig_plan)}')

    # Part I, find the volume of the hole
    hole_volume = dig_hole([(d, l) for d, l, _ in dig_plan])
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 18 part 1, is: '
                              f'{hole_volume}')

    # Part II, find the true volume of the hole
    true_dig_plan = unpack_dig_plan(dig_plan)
    true_hole_volume = dig_hole(true_dig_plan)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 18 part 2, is: '
                              f'{true_hole_volume}')


if __name__ == '__main__':
    main()

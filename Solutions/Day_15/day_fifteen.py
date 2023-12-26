"""Day Fifteen of Advent of Code 2023."""

# Import required packages
from typing import List, Tuple, Dict
import logging


# Parameters
input_file = 'input.txt'


# Constants
input_delimiter = ','
remove_symbol = '-'
add_symbol = '='
num_lens_boxes = 256


# IO
def load_initialization_sequence() -> List[str]:
    """Load the initialization sequence from the input file."""
    logging.log(logging.DEBUG, f'Loading initialization sequence from {input_file}.')
    with open(input_file, 'r') as f:
        initialization_sequence = f.read().strip().split(input_delimiter)
    return initialization_sequence


# Hash described in the problem
def given_hash(input_str: str) -> int:
    """Hash the given input string as described in the problem."""
    hash_out = 0
    for c in input_str:
        c_ascii = ord(c)
        hash_out += c_ascii
        hash_out *= 17
        hash_out = hash_out % num_lens_boxes
    return hash_out


def calculate_lens_sequence(initialization_sequence: List[str]) -> Dict[int, List[Tuple[str, int]]]:
    """Calculate the lens sequence from the initialization sequence.

    Returns a dictionary with keys the box ids and values a list of lens. Each lens is a tuple of the label and the
    focusing power.
    """
    # Setup lens sequence dictionary
    lens_sequence = {idx: [] for idx in range(num_lens_boxes)}

    # Loop through instructions
    for instruction in initialization_sequence:

        # Parse the instruction determining the case we are in
        if remove_symbol in instruction:
            # Removing the lens (if it exists)
            rm_label = instruction.split(remove_symbol)[0]
            box_id = given_hash(rm_label)
            box_sequence = lens_sequence[box_id]

            # Is the lens present in the box sequence?
            lens_idx = None
            for idx, (curr_label, _) in enumerate(box_sequence):
                if rm_label == curr_label:
                    lens_idx = idx
                    break  # No duplicate labels by construction

            # If the lens is present, remove it. Otherwise, do nothing
            if lens_idx is not None:
                del box_sequence[lens_idx]

        elif add_symbol in instruction:
            # Adding the lens
            add_label, focus = instruction.split(add_symbol)
            focus = int(focus)
            box_id = given_hash(add_label)
            box_sequence = lens_sequence[box_id]

            # Is the lens present in the box sequence?
            lens_idx = None
            for idx, (curr_label, _) in enumerate(box_sequence):
                if add_label == curr_label:
                    lens_idx = idx
                    break

            # If the lens is present, update its focus
            if lens_idx is not None:
                box_sequence[lens_idx] = (add_label, focus)
            # Otherwise, add lens to the end of this box's sequence
            else:
                box_sequence.append((add_label, focus))

        else:
            raise ValueError(f'Instruction {instruction} not recognized.')

    return lens_sequence


def calculate_focusing_power(lens_sequence: Dict[int, List[Tuple[str, int]]]) -> int:
    """Calculate the total focusing power of the lens sequence."""
    logging.log(logging.DEBUG, f'Calculating the total focusing power of the lens sequence.')

    # Focusing power is determined at the lens level, so loop through the lens sequences
    total_focusing_power = 0
    for box_idx in range(num_lens_boxes):
        box_sequence = lens_sequence[box_idx]
        for slot_num_minus_one, (_, focus) in enumerate(box_sequence):
            total_focusing_power += (box_idx + 1) * (slot_num_minus_one + 1) * focus  # Defined in the problem
    return total_focusing_power


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    initialization_sequence = load_initialization_sequence()

    # Part I, hash and sum
    initialization_hash_s = [given_hash(initialization) for initialization in initialization_sequence]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 15 part 1, is: '
                              f'{sum(initialization_hash_s)}')

    # Part II, build the lens sequence, calculate the total power
    lens_sequence = calculate_lens_sequence(initialization_sequence)
    total_focusing_power = calculate_focusing_power(lens_sequence)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 15 part 2, is: '
                              f'{total_focusing_power}')


if __name__ == '__main__':
    main()

"""Day Nine of Advent of Code 2023."""

# Import required packages
import logging
from typing import List
import numpy as np

# Parameters
input_file = 'input.txt'


# IO
def load_sequences() -> List[List[int]]:
    """Load the sequences from the input file."""
    with open(input_file, 'r') as f:
        return [[int(num) for num in line.strip().split()] for line in f.readlines()]


# Processing
def extrapolate_sequence(sequence: List[int]) -> int:
    """Calculate the next value in the sequence."""
    sequence = np.array(sequence)
    last_values = [sequence[-1]]
    while not np.all(sequence == 0):
        sequence = np.diff(sequence)
        last_values.append(sequence[-1])
    return sum(last_values)


def reverse_extrapolate_sequence(sequence: List[int]) -> int:
    """Calculate the value before the first value in the sequence."""
    sequence = np.array(sequence)
    first_values = [sequence[0]]
    while not np.all(sequence == 0):
        sequence = np.diff(sequence)
        first_values.append(sequence[0])

    # signatures of values in sum are alternating signs
    # first value is positive, second is negative, etc.
    signatures = [-1 if idx % 2 else 1 for idx in range(len(first_values))]
    signatures = np.array(signatures)
    return sum(first_values * signatures)

def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    sequences = load_sequences()
    logging.log(logging.DEBUG, f'Loaded {len(sequences)} sequences.')

    # Get extrapolated values, the sum of which is the answer to the first part
    extrapolated_values = [extrapolate_sequence(sequence) for sequence in sequences]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 9 part 1, is: {sum(extrapolated_values)}')

    # Get reverse extrapolated values, the sum of which is the answer to the second part
    reverse_extrapolated_values = [reverse_extrapolate_sequence(sequence) for sequence in sequences]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 9 part 2, is: '
                              f'{sum(reverse_extrapolated_values)}')


if __name__ == '__main__':
    main()

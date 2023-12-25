"""Day One of Advent of Code 2023."""

# Import required packages
import numpy as np
from typing import List
import logging
import re

# Parameters
input_file = 'input.txt'
test_input_file = 'test_data.txt'

# Constants
digits = [str(x) for x in range(10)]


# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [np.array(list(line.strip())) for line in f.readlines()]


# Data processing
def get_calibration_values(input_data: List[str]) -> List[int]:
    """Get the calibration values."""
    # Filter out everything that isn't a digit
    logging.log(logging.DEBUG, 'Filtering out non-digits')
    digit_only_data = [line[np.isin(line, digits)] for line in input_data]

    # Get the calibration values
    logging.log(logging.DEBUG, 'Calculating calibration values')
    first_digit = [line[0] for line in digit_only_data]
    last_digit = [line[-1] for line in digit_only_data]
    calibration_values = [int(first+last) for first, last in zip(first_digit, last_digit)]

    return calibration_values


def insert_word_digits(input_data: List[str]) -> List[str]:
    """Replace the first letter in a digit word with the corresponding digit.

    E.g. 'one' becomes '1ne'
    """
    # Define the mapping
    mapping = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
               'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}

    # Copy to avoid side effects
    digit_inserted_data = input_data.copy()

    # Convert the words to digits
    logging.log(logging.DEBUG, 'Converting words to digits')
    for idx, line in enumerate(digit_inserted_data):
        # Convert to string to allow use of string methods
        line = ''.join(line)

        # Find positions to be replaced with digits
        replacement_positions = {}
        for word, digit in mapping.items():
            positions_word_appears_matchs = re.finditer(word, line)
            positions_word_appears = [match.start() for match in positions_word_appears_matchs]
            if len(positions_word_appears) > 0:
                replacement_positions[digit] = positions_word_appears

        # Convert back to array for easy updating
        line = np.array(list(line))
        # Replace the letters with digits
        for digit, positions in replacement_positions.items():
            line[np.array(positions)] = digit

        # Convert back to array for updating the data
        digit_inserted_data[idx] = line

    return digit_inserted_data

def main():
    """Solve the puzzle."""
    logging.basicConfig(level=logging.INFO)

    # Load the input data
    input_data = load_input()

    # Find the sum of the calibration values (answer to part 1)
    calibration_values = get_calibration_values(input_data)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 1 part 1, is: {sum(calibration_values)}')

    # Replace the first letter in a digit word with the corresponding digit.
    # E.g. 'one' becomes '1ne'
    # This is done so that when words overlap, both are converted to digits in the correct order.
    # Since no digits are substrings of other digits, this is a safe operation as long as the replacements are done
    # simultaneously not sequentially.
    digit_inserted_data = insert_word_digits(input_data)

    # Find the sum of the correct calibration values (answer to part 2)
    calibration_values = get_calibration_values(digit_inserted_data)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 1 part 2, is: {sum(calibration_values)}')


if __name__ == "__main__":
    main()

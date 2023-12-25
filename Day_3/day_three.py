"""Day Three of Advent of Code 2023."""

# Import required packages
import numpy as np
import logging

# Parameters
input_file = 'input.txt'

# Constants
digits = [str(x) for x in range(10)]
dot = '.'
gear = '*'


# IO
def load_input() -> np.ndarray:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return np.array([list(line.strip()) for line in f.readlines()])


# Data processing
def extract_part_number_mask(input_data: np.ndarray) -> np.ndarray:
    """Extract a mask indicating where the part numbers are."""
    # Identify which characters are digits
    logging.log(logging.DEBUG, 'Masking digits')
    digit_mask = np.isin(input_data, digits)

    # Identify which characters are next to a symbol
    # First pad array with dot values
    logging.log(logging.DEBUG, 'Masking adjacent symbols')
    padded_data = np.pad(input_data, 1, constant_values=dot)
    symbol_mask = np.bitwise_not(np.isin(padded_data, digits + [dot]))
    adjacent_symbol_mask = symbol_mask[:-2, :-2] | symbol_mask[1:-1, :-2] | symbol_mask[2:, :-2] | \
                           symbol_mask[:-2, 1:-1] | symbol_mask[2:, 1:-1] | symbol_mask[:-2, 2:] | \
                           symbol_mask[1:-1, 2:] | symbol_mask[2:, 2:]

    # Combine masks
    logging.log(logging.DEBUG, 'Finding initial part mask')
    part_mask = np.bitwise_and(digit_mask, adjacent_symbol_mask)

    # Infectious spread of the part number mask to the whole part number
    logging.log(logging.DEBUG, 'Spreading part mask to whole part number')
    new_part_mask = part_mask
    part_mask = np.zeros_like(part_mask, dtype=bool)
    while not np.all(new_part_mask == part_mask):
        part_mask = new_part_mask
        padded_part_mask = np.pad(part_mask, ((0, 0), (1, 1)), constant_values=False)
        element_to_left = padded_part_mask[:, :-2]
        element_to_right = padded_part_mask[:, 2:]
        new_part_mask = (part_mask | element_to_left | element_to_right) & digit_mask

    return part_mask

def extract_part_numbers(input_data: np.ndarray) -> list:
    """Extract the part numbers from the input data."""
    # Find all the part numbers
    part_mask = extract_part_number_mask(input_data)

    # Extract numbers by searching through the part mask
    logging.log(logging.DEBUG,'Extracting part numbers')
    part_numbers = []
    for row in range(part_mask.shape[0]):
        in_part = False
        for col, is_part in enumerate(part_mask[row, :]):
            if is_part:
                if in_part:
                    part_numbers[-1] += input_data[row, col]
                else:
                    part_numbers.append(input_data[row, col])
                    in_part = True
            else:
                in_part = False

    # Convert to integers
    part_numbers = [int(part_number) for part_number in part_numbers]
    return part_numbers


def extract_gear_ratios(input_data: np.ndarray) -> list:
    """Extract the gear ratios from the input data."""
    # Find all the gear symbols
    logging.log(logging.DEBUG, 'Finding gear symbols')
    gear_symbol_mask = np.isin(input_data, gear)

    # Find all the part numbers
    logging.log(logging.DEBUG, 'Finding part numbers')
    part_mask = extract_part_number_mask(input_data)

    # Give each individual part number a unique id
    logging.log(logging.DEBUG, 'Giving each part number a unique id')
    part_ids = np.zeros_like(part_mask, dtype=int)  # 0 is reserved for no part
    part_id = 0
    for row in range(part_mask.shape[0]):
        in_part = False
        for col, is_part in enumerate(part_mask[row, :]):
            if is_part:
                if not in_part:
                    in_part = True
                    part_id += 1
                part_ids[row, col] = part_id

            else:
                in_part = False

    # Pad part ids with zeros to avoid edge cases
    part_ids = np.pad(part_ids, 1, constant_values=0)

    # Loop over gear symbols finding which are gears and if so, what their ratios are
    logging.log(logging.DEBUG, 'Extracting gear ratios')
    part_pairs = []
    for row, col in np.argwhere(gear_symbol_mask):
        # Find the part numbers surrounding the potential gear
        surrounding_part_ids = set(part_ids[row:row + 3, col:col + 3].flatten())  # Offset by 1 due to padding
        surrounding_part_ids.remove(0)  # Remove the zero value
        if len(surrounding_part_ids) != 2:  # Not a gear by definition given in question
            continue

        # Add to list the two part ids surrounding the gear
        part_pairs.append(list(surrounding_part_ids))

    # Get part numbers so can convert the part ids to part numbers
    part_numbers = extract_part_numbers(input_data)

    # Convert part ids to part numbers and calculate gear ratios
    logging.log(logging.DEBUG, 'Converting part ids to part numbers')
    gear_ratios = []
    for part_id_1, part_id_2 in part_pairs:
        part_1 = part_numbers[part_id_1 - 1]
        part_2 = part_numbers[part_id_2 - 1]
        gear_ratios.append(part_1 * part_2)

    return gear_ratios


def main():
    """Solve the puzzle."""
    logging.basicConfig(level=logging.INFO)

    # Load the input data
    input_data = load_input()

    # Find part numbers
    part_numbers = extract_part_numbers(input_data)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 3 part 1, is: {sum(part_numbers)}')

    # Find the gears and their ratios
    gear_ratios = extract_gear_ratios(input_data)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 3 part 2, is: {sum(gear_ratios)}')


if __name__ == "__main__":
    main()

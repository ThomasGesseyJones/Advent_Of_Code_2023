"""Day Four of Advent of Code 2023."""

# Import required packages
import numpy as np
import logging
from typing import List, Tuple

# Parameters
input_file = 'input.txt'

# Constants
win_elfs_divider = '|'
card_prefix = 'Card '
card_suffix = ':'


# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Data processing
def extract_all_cards_info(input_data: List[str]) -> Tuple[np.array, np.array, np.array]:
    """Extract the key card information from the input data."""
    logging.log(logging.DEBUG, 'Extracting game data')

    # Treat the first line as a special case as need it to initialize the arrays
    first_line = input_data[0]
    card_id, card_win_numbers, card_elfs_numbers = extract_one_cards_info(first_line)

    # Initialize the arrays
    all_cards_ids = np.zeros(len(input_data), dtype=int)
    all_winning_numbers = np.zeros((len(input_data), len(card_win_numbers)), dtype=int)
    all_elfs_numbers = np.zeros((len(input_data), len(card_elfs_numbers)), dtype=int)

    # Add the first line information to the arrays
    all_cards_ids[0] = card_id
    all_winning_numbers[0, ...] = card_win_numbers
    all_elfs_numbers[0, ...] = card_elfs_numbers

    # Loop over the remaining lines
    for idx, line in enumerate(input_data[1:]):
        card_id, card_win_numbers, card_elfs_numbers = extract_one_cards_info(line)
        all_cards_ids[idx+1] = card_id
        all_winning_numbers[idx+1, ...] = card_win_numbers
        all_elfs_numbers[idx+1, ...] = card_elfs_numbers

    return all_cards_ids, all_winning_numbers, all_elfs_numbers


def extract_one_cards_info(input_line) -> Tuple[int, np.array, np.array]:
    """Extract the card data from an input line."""
    card_id, numbers = input_line.split(card_suffix)
    card_id = int(card_id[len(card_prefix):])
    win_numbers, elfs_numbers = numbers.split(win_elfs_divider)
    win_numbers = np.array([int(x) for x in win_numbers.split()])
    elfs_numbers = np.array([int(x) for x in elfs_numbers.split()])
    return card_id, win_numbers, elfs_numbers


def calculate_matches(winning_numbers: np.array, elfs_numbers: np.array) -> np.array:
    """Calculate the number of matches for each card."""
    logging.log(logging.DEBUG, 'Calculating winnings')

    # Find matches for each card
    matches = []
    for card_win_nums, card_elfs_nums in zip(winning_numbers, elfs_numbers):
        matching_numbers = np.sum(np.isin(card_win_nums, card_elfs_nums))
        matches.append(matching_numbers)

    return np.array(matches)


def calculate_number_of_each_card(card_by_card_matches: np.array) -> np.array:
    """Calculate the number of each card, once winnings are accounted for."""
    logging.log(logging.DEBUG, 'Calculating number of each card')

    # Find the number of each card
    # Include the card itself in the winnings
    winnings = np.ones((len(card_by_card_matches) + max(card_by_card_matches)),
                        dtype=int)
    for idx, card_matches in enumerate(card_by_card_matches):
        winnings[idx+1:idx+card_matches+1] += winnings[idx]

    return winnings[:len(card_by_card_matches)]

def main():
    """Solve the puzzle."""
    logging.basicConfig(level=logging.INFO)

    # Load the input data and process into arrays
    input_data = load_input()
    game_ids, winning_numbers, elfs_numbers = extract_all_cards_info(input_data)

    # Find the elves total winnings (answer to part 1)
    card_by_card_matches = calculate_matches(winning_numbers, elfs_numbers)
    logging.log(logging.DEBUG, card_by_card_matches)
    card_by_card_points = np.zeros_like(card_by_card_matches, dtype=int)
    card_by_card_points[card_by_card_matches > 0] = 2 ** (card_by_card_matches[card_by_card_matches > 0] - 1)
    logging.log(logging.DEBUG, np.sum(card_by_card_matches))
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 4 part 1, is: {sum(card_by_card_points)}')

    # Find the elves number of scratchcards (answer to part 2)
    number_of_each_card = calculate_number_of_each_card(card_by_card_matches)
    logging.log(logging.DEBUG, number_of_each_card)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 4 part 2, is: {sum(number_of_each_card)}')



if __name__ == "__main__":
    main()

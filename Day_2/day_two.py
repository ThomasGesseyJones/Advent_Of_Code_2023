"""Day Two of Advent of Code 2023."""

# Import required packages
import numpy as np
from typing import List, Tuple
import logging
import re

# Parameters
input_file = 'input.txt'

# Constants
possible_ball_colors = ['red', 'green', 'blue']
entry_divider = ','
reveal_divider = ';'
header_prefix = 'Game '
header_suffix = ':'


# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Data processing
def extract_game_data(input_data: List[str]) -> List[Tuple]:
    """Extract the game data from the input data.

    Return game data is a list of tuples, where each tuple contains the information for a game.
    The tuple structure is
    (game_id, game_results)
    where game_id is an integer and game_results is a 2D array. Each row in the array
    corresponds to a reveal, and the columns are the revealed balls colors, the entries being the number
    of balls of that color revealed.
    """
    # Extract the game data
    logging.log(logging.DEBUG, 'Extracting game data')
    game_data = []
    for line in input_data:
        # Split the line into the header and the reveals
        header, input_reveals = line.split(header_suffix)
        game_id = int(header[len(header_prefix):])

        # Split the results into the reveals
        input_reveals = input_reveals.split(reveal_divider)

        # Create an array to store the results
        game_results = np.zeros((len(input_reveals), len(possible_ball_colors)), dtype=int)

        # Populate the results array
        for reveal_idx, input_reveals in enumerate(input_reveals):
            entries = input_reveals.split(entry_divider)
            for entry in entries:
                for c_idx, color in enumerate(possible_ball_colors):
                    if color in entry:
                        game_results[reveal_idx, c_idx] = int(entry.replace(color, '').strip())
                        break

        # Add the game data to the list
        game_data.append((game_id, game_results))

    return game_data


def find_possible_games(game_data: List[Tuple], ball_counts: dict) -> List[bool]:
    """Find the games that are possible given the ball counts.

    Ball counts is a dictionary with keys corresponding to the ball colors and values corresponding to the
    number of balls of that color. If a ball color is not present in the dictionary, it is assumed that
    there are no balls of that color.
    """
    # Create an array of ball counts for easy comparison
    ball_counts_array = np.array([ball_counts.get(color, 0) for color in possible_ball_colors])
    ball_counts_array = ball_counts_array[np.newaxis, :]  # Comparison trick for comparing with 2D array

    # Find the possible games
    logging.log(logging.DEBUG, 'Finding possible games')
    possible_games = [
        np.all(ball_counts_array >= game_results)
        for _, game_results in game_data
    ]
    return possible_games


def find_minimum_ball_counts(game_data: List[Tuple]) -> List[np.array]:
    """Find the minimum ball counts required for each game."""
    logging.log(logging.DEBUG, 'Finding minimum ball counts')
    minimum_ball_counts = [
        np.max(game_results, axis=0)
        for _, game_results in game_data
    ]
    return minimum_ball_counts

def main():
    """Solve the puzzle."""
    logging.basicConfig(level=logging.INFO)

    # Load the input data
    input_data = load_input()

    # Find the sum of the possible games (answer to part 1)
    game_data = extract_game_data(input_data)
    possible_games = find_possible_games(game_data, {'red': 12, 'green': 13, 'blue': 14})
    possible_game_ids = [game_data[idx][0] for idx, possible in enumerate(possible_games) if possible]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 2 part 1, is: {sum(possible_game_ids)}')

    # Find the sum of the 'power' of the minimum ball counts for eahc game (answer to part 2)
    minimum_ball_counts = np.array(find_minimum_ball_counts(game_data))
    powers = np.prod(minimum_ball_counts, axis=1)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 2 part 2, is: {sum(powers)}')


if __name__ == "__main__":
    main()

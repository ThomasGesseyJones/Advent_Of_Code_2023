"""Day Seventeen of Advent of Code 2023."""

# Import required packages
import logging
from bisect import insort
from typing import List
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from functools import total_ordering


# Parameters
input_file = 'input.txt'


# IO
def load_heat_loss_map(file: str) -> np.ndarray:
    """Load the heat loss map from the input file."""
    logging.log(logging.DEBUG, f'Loading heat loss map from {file}')
    with open(file, 'r') as f:
        heat_loss_map = np.array([[int(x) for x in line.strip()] for line in f.readlines()])
    return heat_loss_map


# Data structures
class Direction(Enum):
    """An enum for the directions."""
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()

    @classmethod
    def orthogonal_directions(cls, direction: 'Direction') -> List['Direction']:
        """Return the orthogonal directions to the given direction."""
        if direction == cls.UP or direction == cls.DOWN:
            return [cls.LEFT, cls.RIGHT]
        else:
            return [cls.UP, cls.DOWN]


@total_ordering
@dataclass
class CrucibleWalker:
    """A random walker through the heat loss map representing a crucible."""
    heat_loss: int
    x: int
    y: int
    end_x: int
    end_y: int
    prev_directions: List[Direction]
    min_move: int = 0
    max_move: int = 3

    # 1-norm distance to the end
    @property
    def distance_to_end(self) -> int:
        """Return the 1-norm distance to the end."""
        return abs(self.x - self.end_x) + abs(self.y - self.end_y)

    # Define the ordering of crucible walkers based on their heat loss and distance to the end
    # We can do this as heat loss is always a positive integer, and the distance to the end is 0 at the end
    def __eq__(self, other: 'CrucibleWalker') -> bool:
        return (self.heat_loss + self.distance_to_end) == (other.heat_loss + other.distance_to_end)

    def __lt__(self, other: 'CrucibleWalker') -> bool:
        return (self.heat_loss + self.distance_to_end) < (other.heat_loss + other.distance_to_end)

    def available_directions(self) -> List[Direction]:
        """Return the directions that are available to move to."""
        if len(self.prev_directions) == 0:
            # If we are at the start, can move any direction
            possible_directions = [Direction.RIGHT, Direction.UP, Direction.DOWN, Direction.LEFT]
        else:
            # Otherwise, the directions our limited by our previous direction and the number of times we have moved
            # in that direction
            number_in_current_direction = 0
            for direction in self.prev_directions[::-1]:
                if direction == self.prev_directions[-1]:
                    number_in_current_direction += 1
                else:
                    break

            # If we have not moved the minimum number of times in the current direction, we can only move in that
            # direction
            if number_in_current_direction < self.min_move:
                possible_directions = [self.prev_directions[-1]]
            # Conversely, if we have moved the maximum number of times in the current direction, we can move in an
            # orthogonal direction
            elif number_in_current_direction == self.max_move:
                possible_directions = Direction.orthogonal_directions(self.prev_directions[-1])
            # Otherwise, we can move in the current direction or an orthogonal direction
            else:
                possible_directions = Direction.orthogonal_directions(self.prev_directions[-1])
                possible_directions.append(self.prev_directions[-1])

        # Remove any directions that would take us out of bounds
        for pos_dir in possible_directions[::-1]:
            if pos_dir == Direction.UP and self.y == 0:
                possible_directions.remove(pos_dir)
            elif pos_dir == Direction.RIGHT and self.x == self.end_x:
                possible_directions.remove(pos_dir)
            elif pos_dir == Direction.DOWN and self.y == self.end_y:
                possible_directions.remove(pos_dir)
            elif pos_dir == Direction.LEFT and self.x == 0:
                possible_directions.remove(pos_dir)

        return possible_directions


def valid_finish(directions: List[Direction], min_move: int) -> bool:
    """Check if the given directions are a valid way to reach the end. E.g. can stop!"""
    if min_move == 0:
        return True

    # Check if we have moved the minimum number of times in the final direction
    last_min_move_directions = directions[-min_move:]
    return len(set(last_min_move_directions)) == 1


def find_minimum_heat_loss(heat_loss_map: np.ndarray,
                           min_move: int = 0,
                           max_move: int = 3,
                           ) -> int:
    """Find minimum heat loss using a modified A* algorithm."""
    logging.log(logging.DEBUG, f'Finding minimum heat loss using a modified A* algorithm')

    # Find our target
    end_y, end_x = heat_loss_map.shape
    end_x -= 1
    end_y -= 1

    # Initialize the list of walkers
    walkers = []
    initial_walker = CrucibleWalker(0, 0, 0, end_x, end_y, [], min_move, max_move)
    walkers.append(initial_walker)

    # To avoid pointless repetition, keep track of which positions a walker has been to already and from which direction
    # and how many tiles it has moved in that direction. Any walker entering the same tile from that direction
    # with equal moves (or more moves providing greater than min_moves) in that direction can be dismissed as duplicate.
    visited_up = np.zeros((end_y + 1, end_x + 1, max_move), dtype=int)
    visited_right = np.zeros((end_y + 1, end_x + 1, max_move), dtype=int)
    visited_down = np.zeros((end_y + 1, end_x + 1, max_move), dtype=int)
    visited_left = np.zeros((end_y + 1, end_x + 1, max_move), dtype=int)
    visited_map = {Direction.UP: visited_up, Direction.RIGHT: visited_right,
                   Direction.DOWN: visited_down, Direction.LEFT: visited_left}

    # Walk through the map
    end_reached_in = np.inf
    while True:
        # Get the current lowest effective heat loss walker (current heat loss + distance to end)
        current_walker = walkers.pop(0)
        logging.log(logging.DEBUG, f'Current walker at ({current_walker.x}, {current_walker.y}) '
                                   f'with effective distance '
                                   f'{current_walker.heat_loss + current_walker.distance_to_end}')

        # See if we reached the end, and can stop
        if current_walker.x == end_x and current_walker.y == end_y and \
                valid_finish(current_walker.prev_directions, min_move):
            logging.log(logging.DEBUG, f'Found a path with heat loss {current_walker.heat_loss}')
            logging.log(logging.DEBUG, f'Path: {current_walker.prev_directions}')
            return current_walker.heat_loss

        # Find possible directions
        possible_directions = current_walker.available_directions()

        # Create walkers in each direction
        new_walkers = []
        for possible_dir in possible_directions:
            # Get new position
            new_x = current_walker.x
            new_y = current_walker.y

            if possible_dir == Direction.UP:
                new_y -= 1
            elif possible_dir == Direction.RIGHT:
                new_x += 1
            elif possible_dir == Direction.DOWN:
                new_y += 1
            elif possible_dir == Direction.LEFT:
                new_x -= 1
            else:
                raise ValueError(f'Unknown direction {possible_dir}')


            # Skip if duplicate, if not update the visitation record
            num_in_direction = 1
            for prev_dir in current_walker.prev_directions[::-1]:
                if prev_dir == possible_dir:
                    num_in_direction += 1
                else:
                    break

            visit_map = visited_map[possible_dir]
            if visit_map[new_y, new_x, num_in_direction - 1]:
                continue
            else:
                if num_in_direction >= min_move:
                    visit_map[new_y, new_x, (num_in_direction - 1):] = 1
                else:
                    visit_map[new_y, new_x, num_in_direction - 1] = 1

            # Otherwise, create the new walker unless it is already worse than the best solution
            new_heat_loss = current_walker.heat_loss + heat_loss_map[new_y, new_x]
            if new_heat_loss > end_reached_in:
                continue
            new_walker = CrucibleWalker(new_heat_loss, new_x, new_y,  end_x, end_y,
                                        current_walker.prev_directions + [possible_dir],
                                        min_move, max_move)
            new_walkers.append(new_walker)

        # Add the new walkers to the list
        for new_walker in new_walkers:
            insort(walkers, new_walker)

        # Check if any of the new walkers reached the end, as then can dismiss any walkers with higher heat loss
        for new_walker in new_walkers:
            if new_walker.x == end_x and new_walker.y == end_y:
                # Check if valid way to reach a stop, e.g., minimum number of moves in the final direction
                if not valid_finish(new_walker.prev_directions, min_move):
                    continue

                # If so, have reached the end with this heat loss
                end_reached_in = min(new_walker.heat_loss, end_reached_in)
                for idx in range(len(walkers)):
                    if walkers[idx].heat_loss > end_reached_in:
                        walkers = walkers[:idx]  # Remove all walkers with higher effective heat loss
                        break


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    heat_loss_map = load_heat_loss_map(input_file)
    logging.log(logging.DEBUG, f'Loaded heat loss map with shape {heat_loss_map.shape}')

    # Part I, find the minimum heat loss
    minimum_heat_loss = find_minimum_heat_loss(heat_loss_map)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 17 part 1, is: '
                              f'{minimum_heat_loss}')

    # Part II, find the minimum heat loss, now with ultra crucibles!
    minimum_heat_loss = find_minimum_heat_loss(heat_loss_map, 4, 10)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 17 part 2, is: '
                              f'{minimum_heat_loss}')


if __name__ == '__main__':
    main()

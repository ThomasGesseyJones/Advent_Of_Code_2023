"""Day Twenty Three of Advent of Code 2023."""

# Import required packages
from __future__ import annotations
import numpy as np
import logging
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum, auto
from typing import List

# Parameters
input_file = 'input.txt'


# Constants
start_row = 0
start_col = 1

start_label = 'S'
end_label = 'E'

forest_tile = '#'
path_tile = '.'
right_only_tile = '>'
left_only_tile = '<'
up_only_tile = '^'
down_only_tile = 'v'
one_way_tiles = [right_only_tile, left_only_tile, up_only_tile, down_only_tile]


# IO
def load_input(file_path: str) -> np.ndarray:
    """Load the input map of forest."""
    logging.log(logging.DEBUG, f'Loading input file: {file_path}')
    map_of_forest = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            map_of_forest.append(list(line))
    return np.array(map_of_forest)


# Data processing
class Direction(Enum):
    """Enum for the direction a walker is facing."""
    right = auto()
    left = auto()
    up = auto()
    down = auto()

    @classmethod
    def opposite(cls, direction: 'Direction') -> 'Direction':
        """Return the opposite direction."""
        if direction == Direction.right:
            return Direction.left
        elif direction == Direction.left:
            return Direction.right
        elif direction == Direction.up:
            return Direction.down
        elif direction == Direction.down:
            return Direction.up
        else:
            raise ValueError(f'Unknown direction: {direction}')


direction_to_delta = {
    Direction.right: (0, 1),
    Direction.left: (0, -1),
    Direction.up: (-1, 0),
    Direction.down: (1, 0)
}


def convert_map_to_network(map_of_forest: np.ndarray) -> nx.DiGraph:
    """Convert the map to a directed network."""
    logging.log(logging.DEBUG, f'Converting map to network.')

    # Create a network
    map_as_network = nx.DiGraph()
    map_as_network.add_node((start_row, start_col))  # Start node
    map_as_network.add_node((map_of_forest.shape[0] - 1, map_of_forest.shape[1] - 2))  # End node

    # Walk through the map adding nodes and edges as we go
    walkers = [(start_row, start_col, Direction.down, start_row, start_col)]
    while len(walkers) > 0:
        current_walker = walkers.pop(0)
        row, col, direction, came_from_row, came_from_col = current_walker

        # Follow walker until we hit a node, e.g., a place with more than one path from it, or to it,
        # or we hit a dead end
        one_way = False
        dead_end = False
        steps = 0
        possible_new_walkers = []
        while True:
            # Take a step in the current direction
            steps += 1
            delta = direction_to_delta[direction]
            row += delta[0]
            col += delta[1]

            # Was the step one way?
            if map_of_forest[row, col] in one_way_tiles:
                one_way = True

            # Check if we are at the end of the map
            if row == map_of_forest.shape[0] - 1 and col == map_of_forest.shape[1] - 2:
                logging.log(logging.DEBUG, f'Found the end of the map.')
                break

            # If not, see our options for a next step
            possible_directions = [Direction.right, Direction.left, Direction.up, Direction.down]
            possible_directions.remove(Direction.opposite(direction))  # Can't go back the way we came
            possible_new_walkers = []
            for possible_direction in possible_directions:
                # Take a virtual step in the possible direction
                delta = direction_to_delta[possible_direction]
                next_row = row + delta[0]
                next_col = col + delta[1]

                # Don't go off the map
                if (next_row < 0 or next_row >= map_of_forest.shape[0] or next_col < 0 or
                        next_col >= map_of_forest.shape[1]):
                    continue

                # Can we go that way? Or could we have come from that way (symbolized with virtual walker with
                # None direction)? We need to include the latter to handle nodes with only one way out but multiple
                # ways in
                if map_of_forest[next_row, next_col] == path_tile:
                    possible_new_walkers.append((row, col, possible_direction))
                elif possible_direction == Direction.right and map_of_forest[next_row, next_col] == right_only_tile:
                    possible_new_walkers.append((row, col, possible_direction))
                elif possible_direction == Direction.right and map_of_forest[next_row, next_col] == left_only_tile:
                    possible_new_walkers.append((row, col, None))
                elif possible_direction == Direction.left and map_of_forest[next_row, next_col] == left_only_tile:
                    possible_new_walkers.append((row, col, possible_direction))
                elif possible_direction == Direction.left and map_of_forest[next_row, next_col] == right_only_tile:
                    possible_new_walkers.append((row, col, None))
                elif possible_direction == Direction.up and map_of_forest[next_row, next_col] == up_only_tile:
                    possible_new_walkers.append((row, col, possible_direction))
                elif possible_direction == Direction.up and map_of_forest[next_row, next_col] == down_only_tile:
                    possible_new_walkers.append((row, col, None))
                elif possible_direction == Direction.down and map_of_forest[next_row, next_col] == down_only_tile:
                    possible_new_walkers.append((row, col, possible_direction))
                elif possible_direction == Direction.down and map_of_forest[next_row, next_col] == up_only_tile:
                    possible_new_walkers.append((row, col, None))

            # How many options do we have?
            if len(possible_new_walkers) == 0:
                # Dead end, so abandon this walker
                dead_end = True
                break
            elif len(possible_new_walkers) > 1:
                # More, then one choice (or way in), we have hit a node
                break

            # Otherwise, we have one choice, so keep going
            direction = possible_new_walkers[0][2]

        # If we hit a dead end, then abandon this walker
        if dead_end:
            continue

        # Otherwise, need to add edges, and potentially a new node and walkers
        # Start by seeing is node already exists
        had_node = map_as_network.has_node((row, col))
        if not had_node:
            # Add the node
            map_as_network.add_node((row, col))

        # Add the edge(s)
        if one_way:
            map_as_network.add_edge((came_from_row, came_from_col), (row, col), weight=steps)
        else:
            map_as_network.add_edge((came_from_row, came_from_col), (row, col), weight=steps)
            map_as_network.add_edge((row, col), (came_from_row, came_from_col), weight=steps)

        # Add new walkers leaving the node if node never reached before
        if not had_node:
            for possible_new_walker in possible_new_walkers:
                if possible_new_walker[2] is not None:  # Don't add virtual walkers, they are just flags
                    walkers.append((*possible_new_walker, row, col))

    # Rename the start and end nodes
    map_as_network = nx.relabel_nodes(map_as_network,
                                      {(start_row, start_col): start_label,
                                       (map_of_forest.shape[0] - 1, map_of_forest.shape[1] - 2): end_label})

    # Should have built the network
    return map_as_network


def find_max_path_length(map_as_network: nx.DiGraph | nx.Graph, greedy_penultimate_node: bool = False) -> int:
    """Find the maximum length of a simple path through the network.

    Can be done more compactly by using networkx builtin function, all_simple_paths, but I wanted to have a go at doing
    it myself.

    If greedy_penultimate_node is True, then the penultimate node is assumed to be greedy, and the path is completed
    greedily. This is a problem-specific optimization.
    """
    logging.log(logging.DEBUG, f'Finding path lengths.')

    # Get start node information and the one we can reach from it (should only be one)
    start_node = start_label
    assert len(list(map_as_network.neighbors(start_node))) == 1
    next_node = list(map_as_network.neighbors(start_node))[0]
    end_node = end_label

    # Get the penultimate node (used for a problem-specific greedy optimization)
    penultimate_node = None
    if greedy_penultimate_node:
        if isinstance(map_as_network, nx.DiGraph):
            penultimate_nodes = list(map_as_network.predecessors(end_node))
        else:
            penultimate_nodes = list(map_as_network.neighbors(end_node))
        assert len(penultimate_nodes) == 1
        penultimate_node = penultimate_nodes[0]

    # Function to find the length of a path through the network
    def path_length(path_list: List[str]) -> int:
        """Find the length of a path through the network."""
        return sum([map_as_network.get_edge_data(node1, node2)['weight'] for node1, node2 in
                    zip(path_list[:-1], path_list[1:])])

    # Loop over paths using iterators
    path_iterators = [start_node, next_node, map_as_network.neighbors(next_node)]
    path = [start_node, next_node]
    max_path_length = -1
    found_paths = 0 # For progress logging
    output_value = False
    while len(path_iterators) > 2:
        if found_paths % 100_000 == 0 and not output_value:
            logging.log(logging.DEBUG, f'Found {found_paths} paths.')
            output_value = True
        elif found_paths % 100_000 == 1:
            output_value = False

        # Get the current path iterator and the current node
        current_path_iterator = path_iterators[-1]
        try:
            next_node = next(current_path_iterator)
        except StopIteration:
            path.pop()  # Remove the last node from the path as we have considered all its options
            path_iterators.pop()  # Remove the used-up iterator
            continue

        # Skip the node if we have already visited it
        if next_node in path:
            continue

        # If the penultimate node is greedy, then complete the path greedily and stop
        if greedy_penultimate_node & (next_node == penultimate_node):
            max_path_length = max(max_path_length, path_length(path + [next_node, end_node]))
            found_paths += 1
            continue
        # Otherwise, stop only when we have already completed the path
        elif next_node == end_node:
            max_path_length = max(max_path_length, path_length(path))
            found_paths += 1
            continue

        # Otherwise, we go deeper
        path.append(next_node)
        path_iterators.append(nx.neighbors(map_as_network, next_node))

    # Return the maximum path length
    logging.log(logging.DEBUG, f'Found {found_paths} paths.')
    return  max_path_length


def remove_directivity(directed_map_as_network: nx.DiGraph) -> nx.Graph:
    """Remove the directivity from the graph"""
    logging.log(logging.DEBUG, f'Removing directivity from the graph.')

    # Make a new network
    map_as_network = nx.Graph()

    # Add the nodes
    map_as_network.add_nodes_from(directed_map_as_network.nodes())

    # Add the edges and their weights
    for edge in directed_map_as_network.edges():
        if edge not in map_as_network.edges():
            map_as_network.add_edge(*edge, weight=directed_map_as_network.get_edge_data(*edge)['weight'])

    # Return the non-directed map
    return map_as_network


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)
    draw_network = False

    # Load the input
    map_of_forest = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded a map of forest with shape: {map_of_forest.shape}')

    # Part I, find the longest path through the forest
    map_as_network = convert_map_to_network(map_of_forest)
    if draw_network:
        nx.draw(map_as_network, with_labels=True)
        plt.show()
    directed_max_path_length = find_max_path_length(map_as_network, greedy_penultimate_node=True)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 23 part 1, is: '
                                f'{directed_max_path_length}')

    # Part II, find the longest path through the forest, ignoring the one-way system.
    undirected_map_as_network = remove_directivity(map_as_network)
    if draw_network:
        nx.draw(undirected_map_as_network, with_labels=True)
        plt.show()
    undirected_max_path_length = find_max_path_length(undirected_map_as_network, greedy_penultimate_node=True)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 23 part 2, is: '
                                f'{undirected_max_path_length}')


if __name__ == '__main__':
    main()

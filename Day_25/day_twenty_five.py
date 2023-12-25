"""Day Twenty Five of Advent of Code 2023."""

# Import required packages
import numpy as np
import logging
import networkx as nx
from typing import List, Set

# Parameters
input_file = 'input.txt'


# Constants
component_other_components_list_delimiter = ':'
components_list_separator = ' '


# IO
def load_input(file_path: str) -> nx.Graph:
    """Load the component network from the input file."""
    logging.log(logging.DEBUG, f'Loading input file: {file_path}')
    component_network = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            source_component, target_components = line.split(component_other_components_list_delimiter)
            source_component = source_component.strip()
            for target_component in target_components.split(components_list_separator):
                if target_component.strip() == '':
                    continue
                component_network.add_edge(source_component, target_component)
    return component_network


def partitioning_by_minimal_cut(network: nx.Graph, known_minimal_cut_size: int=None) -> List[Set]:
    """Find the minimal cut of the network, and return the partition of the network that remains.

    Minimal cut found using the Ford-Fulkerson algorithm implemented in networkx. It's Christmas, I am not going to
    implement that myself :)
    """
    logging.log(logging.DEBUG, f'Finding the minimal cut of the network')

    # Source can be arbitrary, so we pick the first node in the network. We will try all possible target nodes
    # to find the minimal cut later.
    network_nodes = list(network.nodes)
    source_node = network_nodes[0]

    # For networkx Ford-Fulkerson algorithm to work, we need to define a capacity for each edge. Since we are interested
    # in the minimal cut in terms of wire number (edge number), we set the capacity of each edge to 1.
    for edge in network.edges:
        network.edges[edge]['capacity'] = 1

    # Loop over all possible target nodes, and find the minimal cut
    minimal_cut_size = np.inf
    minimal_cut_partition = None
    for target_node in network_nodes[1:]:
        # Cut for that source, target pair
        source_target_min_cut_value, partition = nx.minimum_cut(network, source_node, target_node)
        if source_target_min_cut_value < minimal_cut_size:
            minimal_cut_size = source_target_min_cut_value
            minimal_cut_partition = partition

            # If we know the minimal cut size, we can stop as soon as we find a cut of that size
            if known_minimal_cut_size is not None and minimal_cut_size == known_minimal_cut_size:
                break

    return minimal_cut_partition


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    component_network = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded a component network with {component_network.number_of_nodes()} nodes and '
                                 f'{component_network.number_of_edges()} edges')

    # Part I, find the three wires we need to remove to disconnect the network, and the size of the partition
    # that remains
    minimal_cut_partition = partitioning_by_minimal_cut(component_network, known_minimal_cut_size=3)
    partition_sizes = [len(partition) for partition in minimal_cut_partition]
    product_of_partition_sizes = np.prod(partition_sizes)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 25 part 1, is: '
                                f'{product_of_partition_sizes}')

    # Part II is not missing, it just doesn't require any code!
    logging.log(logging.INFO, f'Merry Christmas!')


if __name__ == '__main__':
    main()

"""Day Twenty Two of Advent of Code 2023."""

# Import required packages
import numpy as np
import logging
from typing import List
from dataclasses import dataclass
import networkx as nx


# Parameters
input_file = 'input.txt'


# Constants
start_end_split = '~'
coordinate_delimiter = ','


# Data structures
@dataclass
class Block:
    """A sand block."""
    min_x_idx: int
    max_x_idx: int
    min_y_idx: int
    max_y_idx: int
    min_z_idx: int
    max_z_idx: int

    settled: bool = False
    on_top_of: List['Block'] = None
    supporting: List['Block'] = None

    def add_on_top_of(self, block: 'Block'):
        """Add a block to the list of blocks this block is on top of."""
        if self.on_top_of is None:
            self.on_top_of = []
        self.on_top_of.append(block)

    def add_supporting(self, block: 'Block'):
        """Add a block to the list of blocks this block is supporting."""
        if self.supporting is None:
            self.supporting = []
        self.supporting.append(block)

    @property
    def num_supporting(self) -> int:
        """Return the number of blocks this block is supporting."""
        if self.supporting is None:
            return 0
        else:
            return len(self.supporting)

    @property
    def num_on_top_of(self) -> int:
        """Return the number of blocks this block is on top of."""
        if self.on_top_of is None:
            return 0
        else:
            return len(self.on_top_of)

    def __str__(self):
        """Return a string representation of the block."""
        return f'({self.min_x_idx}, {self.min_y_idx}, {self.min_z_idx}) -> ({self.max_x_idx}, {self.max_y_idx}, {self.max_z_idx})'

    def __repr__(self):
        """Return a string representation of the block."""
        return self.__str__()


# IO
def load_input(file_path: str) -> List[Block]:
    """Load the input file, and extract the block information."""
    logging.log(logging.DEBUG, f'Loading input file: {file_path}')
    with open(file_path, 'r') as f:
        blocks = []
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue

            # Split the line into the start and end coordinates
            start_str, end_str = line.split(start_end_split)
            start_str = start_str.strip()
            end_str = end_str.strip()

            # Split the start and end coordinates into x, y, and z
            start_x, start_y, start_z = start_str.split(coordinate_delimiter)
            end_x, end_y, end_z = end_str.split(coordinate_delimiter)

            # Convert the coordinates to integers
            start_x = int(start_x)
            start_y = int(start_y)
            start_z = int(start_z)
            end_x = int(end_x)
            end_y = int(end_y)
            end_z = int(end_z)

            # Create the block
            block = Block(min_x_idx=start_x, max_x_idx=end_x,
                          min_y_idx=start_y, max_y_idx=end_y,
                          min_z_idx=start_z, max_z_idx=end_z)

            # Add the block to the list of blocks
            blocks.append(block)

        return blocks


# Simulation
def settle_blocks(blocks: List[Block]):
    """Settle blocks under gravity."""
    logging.log(logging.DEBUG, f'Settling blocks')

    # Sort the blocks by their lower z coordinate as settling in this order will ensure the correct result
    blocks = sorted(blocks, key=lambda block: block.min_z_idx)

    # Set up an array to represent the current tower
    max_x = max([block.max_x_idx for block in blocks])
    max_y = max([block.max_y_idx for block in blocks])
    max_z = max([block.max_z_idx for block in blocks])
    tower = np.zeros((max_x + 1, max_y + 1, max_z + 1), dtype=int)

    # Add a ground block to form the base of the tower
    ground_block = Block(min_x_idx=0, max_x_idx=max_x,
                         min_y_idx=0, max_y_idx=max_y,
                         min_z_idx=0, max_z_idx=0)
    ground_block.settled = True
    tower[:, :, 0] = 1

    # Settle the blocks one by one
    settled_blocks = [ground_block]
    while len(blocks) > 0:
        block = blocks.pop(0)

        # Block has either hit the ground, landed on another, or can fall further
        while True:
            # Has the block hit the ground? or another block? These are equivalent due to our use of a ground block
            if np.any(tower[block.min_x_idx:block.max_x_idx + 1,
                            block.min_y_idx:block.max_y_idx + 1,
                            block.min_z_idx - 1]):
                break

            # Block can fall further
            block.min_z_idx -= 1
            block.max_z_idx -= 1

        # Find what the block is on top of as will need to know for disintegrating blocks later
        add_supporting_links(block, settled_blocks)

        # The Block has now settled, we can move onto the next block
        block.settled = True
        settled_blocks.append(block)
        tower[block.min_x_idx:block.max_x_idx+1,
              block.min_y_idx:block.max_y_idx+1,
              block.min_z_idx:block.max_z_idx+1] = 1


def add_supporting_links(block: Block, settled_blocks: List[Block]):
    """Add the supporting links for a block."""
    # Find the blocks that this block is on top of
    for settled_block in settled_blocks:
        # To be on top of a block, the base and top must align
        if not (block.min_z_idx == settled_block.max_z_idx + 1):
            continue

        # Determine if any overlap in x and y
        overlap_x = block.min_x_idx <= settled_block.max_x_idx and block.max_x_idx >= settled_block.min_x_idx
        overlap_y = block.min_y_idx <= settled_block.max_y_idx and block.max_y_idx >= settled_block.min_y_idx
        if overlap_x and overlap_y:
            # Add the supporting link
            settled_block.add_supporting(block)
            block.add_on_top_of(settled_block)


def find_can_be_removed(blocks: List[Block]) -> List[Block]:
    """Find which blocks can be removed."""
    # A block can be removed if for all blocks above it, it is not the only block supporting it
    can_remove_blocks = []
    for block in blocks:
        # Trivial case of not supporting any blocks
        if block.num_supporting == 0:
            can_remove_blocks.append(block)
            continue

        # Otherwise, check each block it is supporting to see if it is the only block it is on top of
        num_supporting = np.array([block.num_on_top_of for block in block.supporting])
        if np.all(num_supporting > 1):
            can_remove_blocks.append(block)

    return can_remove_blocks


def number_that_would_fall_if_removed(blocks: List[Block]) -> List[int]:
    """Find the number of blocks that would fall if each block was removed."""
    logging.log(logging.DEBUG, f'Finding the number of blocks that would fall if each block was removed')

    # The easiest way (but by no means the most efficient way) to do this is to use a directed graph structure and
    # remove nodes. The number of nodes then discontented from the ground node is the number of blocks that would fall
    # if that node was removed.
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(blocks)))
    graph.add_node(-1)  # Represent the ground as a node

    # Add edges between blocks that are on top of each other, pointing from the supporting block to the block on top
    for block in blocks:
        if block.min_z_idx == 1:
            graph.add_edge(-1, blocks.index(block))
        if block.num_supporting == 0:
            continue
        for supporting_block in block.supporting:
            graph.add_edge(blocks.index(block), blocks.index(supporting_block))

    # Find the size of graph normally, e.g., the number of blocks + 1 for the ground
    num_nodes = len(graph.nodes)

    # Try removing each node in turn and see how much the component containing the ground node changes in size
    num_that_would_fall = []
    for block_node in range(len(blocks)):
        # Make a copy of graph to remove the node from, and remove the node from the copy
        temp_graph = graph.copy()
        temp_graph.remove_node(block_node)

        # Find the size of the component containing the ground node
        num_nodes_after = len(nx.descendants(temp_graph, -1)) + 1  # ground node is not a descendant of itself

        # Add the number of nodes that would fall
        num_that_would_fall.append(num_nodes - num_nodes_after - 1)
        # -1 to remove the disintegrated block

    return num_that_would_fall


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    blocks = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded {len(blocks)} blocks')

    # Part I, find the number of blocks that can be safely removed
    settle_blocks(blocks)
    can_be_removed = find_can_be_removed(blocks)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 22 part 1, is: '
                                f'{len(can_be_removed)}')

    # Part II, find the number of blocks that would fall if each block was removed
    num_that_would_fall = number_that_would_fall_if_removed(blocks)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 22 part 2, is: '
                                f'{sum(num_that_would_fall)}')


if __name__ == '__main__':
    main()

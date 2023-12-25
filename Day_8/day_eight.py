"""Day Eight of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Iterator, Tuple, Dict
from itertools import cycle
from math import gcd


# Parameters
input_file = 'input.txt'
right_symbol = 'R'
left_symbol = 'L'
current_next_dividers = '='
redundant_symbols = ['(', ')']
step_split_symbol = ','

start_step = 'AAA'
end_step = 'ZZZ'

parallel_start_step_ending = 'A'
parallel_end_step_ending = 'Z'


# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Data Processing
def extract_step_iterator(input_data) -> Tuple[Iterator, List[str]]:
    """Extract the step direction iterator from the input data."""
    logging.log(logging.DEBUG, f'Extracting step iterator.')
    steps = list(input_data[0])
    step_iterator = cycle(list(steps))
    return step_iterator, steps


def extract_next_step_adjacency_list(input_data) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Extract the (sort of) adjacency list for the next step options and a mapping from node to index."""
    logging.log(logging.DEBUG, f'Extracting next step adjacency list.')
    index_mapping = {}
    adjacency_list = []

    for idx, line in enumerate(input_data[2:]):
        # Get the current step and add to index mapping
        current_step, next_steps = line.split(current_next_dividers)
        current_step = current_step.strip()
        index_mapping[current_step] = idx

        # Get next steps and add to the adjacency list
        next_steps = next_steps.strip()
        for redundant_symbol in redundant_symbols:
            next_steps = next_steps.replace(redundant_symbol, '')
        left_step, right_step = next_steps.split(step_split_symbol)
        left_step = left_step.strip()
        right_step = right_step.strip()
        adjacency_list.append((left_step, right_step))

    return adjacency_list, index_mapping


def calculate_steps_in_path(adjacency_list: List[Tuple[str, str]],
                            index_mapping: Dict[str, int],
                            step_iterator: Iterator) -> int:
    """Calculate the number of steps in the path to escape."""
    steps = 0
    current_node = start_step
    while current_node != end_step:
        steps += 1
        next_move = next(step_iterator)
        if next_move == right_symbol:
            current_node = adjacency_list[index_mapping[current_node]][1]
        elif next_move == left_symbol:
            current_node = adjacency_list[index_mapping[current_node]][0]
        else:
            raise ValueError(f'Unrecognised move {next_move=}')
    return steps


def find_cycle(adjacency_list: List[Tuple[str, str]],
               index_mapping: Dict[str, int],
               steps: list,
               starting_node: str) -> Tuple[int, List]:
     """Find the length and starting point of the limit cycle."""
     logging.log(logging.DEBUG, f'Finding cycle starting at {starting_node=}.')

     # Initialise the path storage
     path_step = (starting_node, 0)

     # Define a new step iterator that also counts the step number in the step pattern
     step_iterator = cycle(zip(list(steps), range(len(steps))))

     # Traverse the graph until we find the cycle
     path = []
     while path_step not in path:
         path.append(path_step)
         next_move, move_id = next(step_iterator)
         if next_move == right_symbol:
             next_node = adjacency_list[index_mapping[path_step[0]]][1]
         elif next_move == left_symbol:
             next_node = adjacency_list[index_mapping[path_step[0]]][0]
         else:
             raise ValueError(f'Unrecognised move {next_move=}')
         path_step = (next_node, move_id)

     # Extract the cycle information
     cycle_start = path.index(path_step)
     cycle_info = path[cycle_start:]  # Called this to avoid confusion with inbuilt cycle function
     return cycle_start, cycle_info


def create_end_node_step_generator(cycle_start: int, cycle_info: List, end_nodes: List[str]) -> Iterator:
    """Convert cycle info into a generator that gives the step number when it is next at an end node."""
    logging.log(logging.DEBUG, f'Creating end node step generator.')
    at_end_node = [ci[0] in end_nodes for ci in cycle_info]
    at_end_idxs = [idx for idx, at_end in enumerate(at_end_node) if at_end]
    logging.log(logging.DEBUG, f'{at_end_idxs=}')
    cycle_length = len(cycle_info)
    def end_node_step_generator(start_idx, length) -> int:
        """Generator that gives the step number when cycle is next at an end node."""
        while True:
            for idx in at_end_idxs:
                yield start_idx + idx
            start_idx += length
    return end_node_step_generator(cycle_start, cycle_length)


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input and extract the key information
    input_data = load_input()
    step_iterator, steps = extract_step_iterator(input_data)
    logging.log(logging.DEBUG, f'{step_iterator=}')
    adjacency_list, index_mapping = extract_next_step_adjacency_list(input_data)
    logging.log(logging.DEBUG, f'{adjacency_list=}')
    logging.log(logging.DEBUG, f'{index_mapping=}')

    # Traverse the graph finding number of steps
    steps_in_path = calculate_steps_in_path(adjacency_list, index_mapping, step_iterator)

    # Output the number of steps taken, the answer to the puzzle
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 8 part 1, is: {steps_in_path}')

    # Brute forcing the ghost path problem is likely not computationally feasible (at least on my laptop!).
    # So will need to find a way to solve it more efficiently. The problem we have is a deterministic chain, and there
    # are a finite number of states equal to the number of nodes times the number of steps in the cycle. We can
    # thus conclude that each path through the states ends up in a cycle with length that is less than the
    # finite number of states. So a good approach would be to first find the length and starting point of the cycle.
    starting_nodes = [node for node in index_mapping.keys() if node.endswith(parallel_start_step_ending)]
    end_nodes = [node for node in index_mapping.keys() if node.endswith(parallel_end_step_ending)]  # Needed later
    cycles = [find_cycle(adjacency_list, index_mapping, steps, starting_node) for starting_node in starting_nodes]

    logging.log(logging.DEBUG, f'Cycle lengths: {[len(c[1]) for c in cycles]}')
    logging.log(logging.DEBUG, f'Cycle starting points: {[c[0] for c in cycles]}')

    # Now let us convert these into generators that give the step number when they are at an
    # end node
    end_node_step_generators = [create_end_node_step_generator(c[0], c[1], end_nodes) for c in cycles]

    # Testing showed that each cycle actually only encounters an end state once in the cycle, so we can simplify
    # the problem greatly by extracting this information. When this first occurs and the length of the cycle.
    first_end_node_step = [next(g) for g in end_node_step_generators]
    cycle_lengths = [len(c[1]) for c in cycles]
    logging.log(logging.DEBUG, f'{first_end_node_step=}')
    logging.log(logging.DEBUG, f'{cycle_lengths=}')

    # Turns out all first end node steps are the same as the cycle lengths, so this boils down to just finding
    # the lowest common multiple of the cycle lengths
    first_match = cycle_lengths[0]
    for cycle_length in cycle_lengths[1:]:
        first_match = first_match * cycle_length // gcd(first_match, cycle_length)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 8 part 2, is: {first_match}')


if __name__ == '__main__':
    main()

"""Day Twelve of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
import tqdm
from functools import lru_cache

# Parameters
input_file = 'input.txt'

# Constants
symbol_operational = '.'
symbol_broken = '#'
symbol_unknown = '?'
info_divider = ' '
block_length_divider = ','


# Data structures
@dataclass
class SpringRecord:
    spring_pattern: str
    blocks: List[int]


# IO
def load_spring_records() -> List[SpringRecord]:
    """Load the spring records from the input file."""
    with open(input_file, 'r') as f:
        records = []
        for line in f.readlines():
            line = line.strip()
            spring_pattern, blocks = line.split(info_divider)
            blocks = [int(b) for b in blocks.split(block_length_divider)]
            records.append(SpringRecord(spring_pattern, blocks))
    return records


# Data processing (blocks are converted to Tuples for use in lru_cache)
@lru_cache(maxsize=1_000)
def no_operational_number_allowed_spring_combinations(spring_pattern: str, blocks: Tuple[int, ...]) -> int:
    """Calculate the allowed spring combinations for a given record providing there are no operational symbols
    in the spring pattern."""
    # Convert blocks to a list, for easier manipulation
    blocks = list(blocks)

    # Simple base cases
    if spring_pattern == '':  # No spring pattern fine if no blocks
        return len(blocks) == 0
    if sum(blocks) + len(blocks) - 1 > len(spring_pattern):  # Too many blocks to fit in
        return 0
    if sum(blocks) < spring_pattern.count(symbol_broken): # Too few blocks for what is already there
        return 0
    if len(blocks) == 0: # No blocks, so no broken symbols allowed
        return spring_pattern.count(symbol_broken) == 0

    # Base case where there is only one block
    if len(blocks) == 1:
        # No constraints so simple solution
        if spring_pattern.count(symbol_broken) == 0:
            return len(spring_pattern) - blocks[0] + 1

        # One broken symbol somewhat limits the allowed block positions
        elif spring_pattern.count(symbol_broken) == 1:
            broken_position = spring_pattern.find(symbol_broken)

            earliest_start = broken_position - blocks[0] + 1
            latest_start = broken_position
            earliest_start = max(earliest_start, 0)  # Ensure we don't go off the start of the string
            latest_start = min(latest_start, len(spring_pattern) - blocks[0])  # Ensure we don't go off the end
            return latest_start - earliest_start + 1

        # Multiple broken symbols further limit the allowed block positions
        else:
            # Only first and last matter, as needs to cover all broken symbols
            first_break_position = spring_pattern.find(symbol_broken)
            last_break_position = spring_pattern.rfind(symbol_broken)

            # Check if the block can fit between the first and last broken symbol
            if last_break_position >= first_break_position + blocks[0]:
                return 0

            # Otherwise, the block can fit between the first and last broken symbol. We just need to count
            # possibilities like above
            earliest_start = last_break_position - blocks[0] + 1
            latest_start = first_break_position
            earliest_start = max(earliest_start, 0)
            latest_start = min(latest_start, len(spring_pattern) - blocks[0])
            return latest_start - earliest_start + 1

    # More than one block left, so we divide and conquer based on the middle block
    middle_block_idx = len(blocks) // 2
    left_blocks = blocks[:middle_block_idx]
    right_blocks = blocks[middle_block_idx + 1:]
    left_spacing = sum(left_blocks) + len(left_blocks)
    right_spacing = sum(right_blocks) + len(right_blocks)

    # Check the chosen block can fit in the gaps left by the constraints either side
    if left_spacing + right_spacing + blocks[middle_block_idx]  > len(spring_pattern):
        return 0

    # Check every place that middle block could go and the corresponding partitioning of the spring pattern
    num_possible_spring_layouts = 0
    for possible_block_start in range(left_spacing, len(spring_pattern) - right_spacing - blocks[middle_block_idx]+1):
        # Check the item to the left and right are not broken as this would mean the block is bigger than it should be
        if possible_block_start > 0:
            if spring_pattern[possible_block_start - 1] == symbol_broken:
                continue  # Not allowed

        possible_block_end = possible_block_start + blocks[middle_block_idx] - 1
        if possible_block_end < len(spring_pattern) - 1:
            if spring_pattern[possible_block_end + 1] == symbol_broken:
                continue # Not allowed

        # Partition the spring pattern based on the chosen block position
        if possible_block_start > 0:
            left_spring_pattern = spring_pattern[:possible_block_start-1]
        else:
            left_spring_pattern = ''

        if possible_block_end < len(spring_pattern) - 1:
            right_spring_pattern = spring_pattern[possible_block_end+2:]
        else:
            right_spring_pattern = ''

        # Add the number of possible layouts for this partitioning to the total
        num_possible_spring_layouts += \
            no_operational_number_allowed_spring_combinations(left_spring_pattern, tuple(left_blocks)) * \
            no_operational_number_allowed_spring_combinations(right_spring_pattern, tuple(right_blocks))

    return num_possible_spring_layouts


@lru_cache(maxsize=1_000)
def number_allowed_spring_combinations(spring_pattern: str, blocks: Tuple[int, ...]) -> int:
    """Calculate the allowed spring combinations for a given record"""
    # Convert blocks to a list, for easier manipulation
    blocks = list(blocks)

    # Base cases
    # Empty case, fine if no blocks
    if spring_pattern == '':
        return len(blocks) == 0

    # Too many blocks to fit in the gaps
    if sum(blocks) > spring_pattern.count(symbol_broken) + spring_pattern.count(symbol_unknown):
        return 0

    # Too few blocks for what is already there
    if sum(blocks) < spring_pattern.count(symbol_broken):
        return 0

    # Simplify the spring pattern if possible
    # Remove repeated operational symbols as they don't change the number of combinations
    if symbol_operational + symbol_operational in spring_pattern:
        old_spring_pattern = ''
        while old_spring_pattern != spring_pattern:
            old_spring_pattern = spring_pattern
            spring_pattern = spring_pattern.replace(symbol_operational + symbol_operational, symbol_operational)

    # Remove operational symbols at the start and end of the string as they don't change the number of combinations
    if spring_pattern[0] == symbol_operational:
        spring_pattern = spring_pattern[1:]
    if spring_pattern[-1] == symbol_operational:
        spring_pattern = spring_pattern[:-1]

    # If no operational symbols left, can't use that as basis for divide and conquer anymore, so use the
    # no_operational_number_allowed_spring_combinations function instead which divides based on blocks of
    # broken symbols
    if spring_pattern.count(symbol_operational) == 0:
        return no_operational_number_allowed_spring_combinations(spring_pattern, tuple(blocks))

    # Divide and conquer, find the operational symbol closest to the middle of the string
    # and split the string there
    operational_positions = np.where(np.array([s for s in spring_pattern]) == symbol_operational)[0]
    mid_point = len(spring_pattern) // 2
    split_point = operational_positions[np.argmin(np.abs(operational_positions - mid_point))]
    left_spring_pattern = spring_pattern[:split_point]
    right_spring_pattern = spring_pattern[split_point + 1:]

    # Consider each case of block partitioning by this split, adding the number of allowed combinations for each
    # partitioning to the total
    number_of_allowed_combinations = 0
    for block_partition in range(len(blocks) + 1):
        left_blocks = blocks[:block_partition]
        right_blocks = blocks[block_partition:]
        left_combinations = number_allowed_spring_combinations(left_spring_pattern, tuple(left_blocks))
        right_combinations = number_allowed_spring_combinations(right_spring_pattern, tuple(right_blocks))
        number_of_allowed_combinations += left_combinations * right_combinations

    return number_of_allowed_combinations


def calculate_number_of_possible_layouts(records: List[SpringRecord]) -> List[int]:
    """Calculate the number of possible layouts for each of set of records"""
    num_possible_layouts = []
    for record in tqdm.tqdm(records):  # Progress bars are fun!
        num_possible_layouts.append(number_allowed_spring_combinations(record.spring_pattern, tuple(record.blocks)))
    return num_possible_layouts


def unfold_records(records: List[SpringRecord]) -> List[SpringRecord]:
    """Unfold records as per description of problem part II."""
    for record in records:
        pattern = record.spring_pattern
        for _ in range(4):
            pattern += symbol_unknown + record.spring_pattern
        record.spring_pattern = pattern

        blocks = record.blocks
        new_blocks = np.tile(blocks, 5)
        record.blocks = new_blocks
    return records


def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    records = load_spring_records()
    logging.log(logging.DEBUG, f'Loaded {len(records)} records')

    # Part I, find the number of possible layouts for each record and take there sum
    num_possible_layouts = calculate_number_of_possible_layouts(records)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 12 part 1, is: '
                              f'{sum(num_possible_layouts)}')

    # Unfold records for part II and repeat
    records = unfold_records(records)
    num_possible_layouts = calculate_number_of_possible_layouts(records)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 12 part 2, is: '
                              f'{sum(num_possible_layouts)}')


if __name__ == '__main__':
    main()

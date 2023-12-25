"""Day Five of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
from dataclasses import dataclass

# Parameters
input_file = 'input.txt'

# Constants
seeds_indicator = 'seeds:'
map_start_indicator = 'map:'

# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Setup data class to represent a map range, a full map, and a map chain
@dataclass
class MapRange:
    """Data class to represent a mapping range."""
    destination_start: int
    source_start: int
    range_length: int

    def in_range(self, source_value: int) -> bool:
        """Check if a source value is in the range."""
        return source_value in range(self.source_start, self.source_start + self.range_length)

    def map_value(self, source_value: int) -> int:
        """Map a source value to a destination value."""
        if self.in_range(source_value):
            return self.destination_start + source_value - self.source_start
        else:
            raise ValueError(f'Source value {source_value} is not in range {self.source_start} to '
                             f'{self.source_start + self.range_length}')


@dataclass
class Map:
    """Data class to represent a mapping."""
    map_ranges: List[MapRange]

    def __post_init__(self):
        """Sort the maps and fill in any gaps."""
        # Sort the map ranges
        self.map_ranges.sort(key=lambda x: x.source_start)

        # Fill in any gaps
        filled_map_ranges = []
        if self.map_ranges[0].source_start > 0:
            filled_map_ranges.append(MapRange(0, 0, self.map_ranges[0].source_start))
        for idx, map_range in enumerate(self.map_ranges[:-1]):
            next_map_range = self.map_ranges[idx + 1]
            if next_map_range.source_start > map_range.source_start + map_range.range_length:
                filled_map_ranges.append(MapRange(map_range.source_start + map_range.range_length,
                                                  map_range.source_start + map_range.range_length,
                                                  next_map_range.source_start - map_range.source_start - map_range.range_length))

        self.map_ranges = filled_map_ranges + self.map_ranges

        # Sort again
        self.map_ranges.sort(key=lambda x: x.source_start)

    def map_value(self, source_value: int) -> int:
        """Map a source value to a destination value."""
        for map_range in self.map_ranges:
            if map_range.in_range(source_value):
                return map_range.map_value(source_value)
        return source_value  # If no mapping range found, return the source value


@dataclass
class MapChain:
    """Data class to represent a mapping chain."""
    maps: List[Map]

    def chain_map_value(self, source_value: int) -> int:
        """Map a source value to the final destination value."""
        step_input = source_value
        for map_step in self.maps:
            step_input = map_step.map_value(step_input)
        return step_input


# Data processing
def extract_seeds_and_maps(input_data: List[str]) -> Tuple[List[int], List[Map]]:
    """Extract the seeds and maps from the input data."""
    # Find the seeds and maps
    logging.log(logging.DEBUG, 'Extracting seeds and maps')

    # Seeds first as at the top of the file
    seeds = [int(x) for x in input_data[0].split(seeds_indicator)[1].strip().split()]

    # Process through building up maps
    maps = []
    current_map_ranges = []
    for line in input_data[1:]:
        if line == '':  # empty line indicates end of map
            if len(current_map_ranges) > 0:
                maps.append(Map(current_map_ranges))
            current_map_ranges = []
            continue
        elif map_start_indicator in line:  # Don't care about the map name yet
            continue
        else:
            destination_start, source_start, range_length = [int(x) for x in line.split()]
            current_map_ranges.append(MapRange(destination_start, source_start, range_length))

    # Add the final map
    if len(current_map_ranges) > 0:
        maps.append(Map(current_map_ranges))

    return seeds, maps


def invert_map_chain(map_chain: MapChain) -> MapChain:
    """Invert a map chain"""
    base_maps = map_chain.maps

    # Invert each map
    inverted_maps = []
    for base_map in base_maps:
        inverted_map_ranges = []
        for base_map_range in base_map.map_ranges:
            inverted_map_ranges.append(MapRange(base_map_range.source_start,
                                                base_map_range.destination_start,
                                                base_map_range.range_length))
        inverted_maps.append(Map(inverted_map_ranges))

    # Reverse the order of the maps
    inverted_maps = inverted_maps[::-1]
    return MapChain(inverted_maps)



# Class to represent a set of seeds now we know they aren't individual values
class SeedSet:
    seed_ranges: List

    def __init__(self, seeds: List[int]):
        """Initialise the seed set."""
        self.seed_ranges = []
        # Input seeds are a list of alternating starts and ranges
        for idx in range(0, len(seeds), 2):
            self.seed_ranges.append(range(seeds[idx], seeds[idx] + seeds[idx + 1]))

    def in_set(self, seed: int) -> bool:
        """Check if a seed is in the set."""
        for seed_range in self.seed_ranges:
            if seed in seed_range:
                return True
        return False



def main():
    """Solve the puzzle."""
    logging.basicConfig(level=logging.INFO)

    # Load the input data and split into seeds and maps
    input_data = load_input()
    seeds, maps = extract_seeds_and_maps(input_data)

    # Build the map chain
    logging.log(logging.DEBUG, 'Building map chain')
    map_chain = MapChain(maps)

    # Evaluate the map chain
    logging.log(logging.DEBUG, 'Evaluating map chain')
    final_values = [map_chain.chain_map_value(seed) for seed in seeds]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 5 part 1, is: {min(final_values)}')

    # Invert the map chain
    logging.log(logging.DEBUG, 'Inverting map chain')
    inverted_map_chain = invert_map_chain(map_chain)

    # Check against the initial seeds
    replicated_seeds = [inverted_map_chain.chain_map_value(final_value) for final_value in final_values]
    assert replicated_seeds == seeds

    # Brute force the answer, we know it is less than the answer to part 1 and so time estimates says checking
    # every value under this will take less than 2 hours. I have to go run some errands, so there is no point in me
    # trying to optimize this as I can just leave it running while I am out.
    logging.log(logging.DEBUG, 'Brute forcing the answer')
    seed_set = SeedSet(seeds)
    location = None
    for location in range(min(final_values) + 1):
        if location % 1000000 == 0:
            logging.log(logging.DEBUG, f'Currently at location {location}')
        seed = inverted_map_chain.chain_map_value(location)
        if seed_set.in_set(seed):
            break

    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 5 part 2, is: '
                              f'{location}')


if __name__ == '__main__':
    main()

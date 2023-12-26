"""Day Ten of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum, auto

# Parameters
input_file = 'input.txt'


# Set-up representation of types of pipes and a point in the pipe network
@dataclass
class PipeType:
    """Representation of the different types of pipes."""
    symbol: str
    north_exit: bool
    east_exit: bool
    south_exit: bool
    west_exit: bool


# All possible pipe types
ground = PipeType(symbol='.', north_exit=False, east_exit=False, south_exit=False, west_exit=False)
vertical_pipe = PipeType(symbol='|', north_exit=True, east_exit=False, south_exit=True, west_exit=False)
horizontal_pipe = PipeType(symbol='-', north_exit=False, east_exit=True, south_exit=False, west_exit=True)
ne_pipe = PipeType(symbol='L', north_exit=True, east_exit=True, south_exit=False, west_exit=False)
nw_pipe = PipeType(symbol='J', north_exit=True, east_exit=False, south_exit=False, west_exit=True)
se_pipe = PipeType(symbol='F', north_exit=False, east_exit=True, south_exit=True, west_exit=False)
sw_pipe = PipeType(symbol='7', north_exit=False, east_exit=False, south_exit=True, west_exit=True)
start_pipe = PipeType(symbol='S', north_exit=True, east_exit=True, south_exit=True, west_exit=True)

pipe_types = [ground, vertical_pipe, horizontal_pipe, ne_pipe, nw_pipe, se_pipe, sw_pipe, start_pipe]


@dataclass
class PipePoint:
    """Representation of a point in the pipe network."""
    x: int
    y: int
    type: PipeType

    def __str__(self):
        return self.type.symbol

    def __repr__(self):
        return self.__str__()

    def coordinates(self) -> Tuple[int, int]:
        """Coordinates of this pipe point."""
        return self.x, self.y

    def positions_connected_to(self) -> List[Tuple[int, int]]:
        """Positions this pipe is connected to."""
        connected_positions = []
        if self.type.north_exit:
            connected_positions.append((self.x, self.y - 1))
        if self.type.east_exit:
            connected_positions.append((self.x + 1, self.y))
        if self.type.south_exit:
            connected_positions.append((self.x, self.y + 1))
        if self.type.west_exit:
            connected_positions.append((self.x - 1, self.y))
        return connected_positions


# IO
def load_pipe_network() -> List[List[PipePoint]]:
    """Load the pipe network from the input file."""
    # Load grid
    with open(input_file, 'r') as f:
        char_grid = [[char for char in line.strip()] for line in f.readlines()]

    # Convert to pipe network
    pipe_network = []
    for y, row in enumerate(char_grid):
        pipe_network.append([])
        for x, char in enumerate(row):
            pipe_network[y].append(PipePoint(x=x, y=y, type=[pipe_type for pipe_type in pipe_types
                                                             if pipe_type.symbol == char][0]))

    return pipe_network


def find_pipe_loop(pipe_network: List[List[PipePoint]]) -> List[PipePoint]:
    """Find the pipe loop hidden in the pipe network."""
    logging.log(logging.DEBUG, f'Finding pipe loop.')

    # Find the start of the pipe loop
    start_point = None
    for y, row in enumerate(pipe_network):
        for x, pipe_point in enumerate(row):
            if pipe_point.type == start_pipe:
                start_point = pipe_point
                break
        if start_point is not None:
            break

    # Start storing the loop
    loop = [start_point]

    # Find the initial direction we are traveling in
    points_connected_to_start = [pipe_network[y][x] for x, y in start_point.positions_connected_to()]
    points_connected_back_to_start = [point for point in points_connected_to_start if start_point.coordinates() in
                                      point.positions_connected_to()]

    # Arbitrarily chose the first direction to go in and start traversing the loop
    previous_point = start_point
    next_point = points_connected_back_to_start[0]
    while next_point != start_point:  # Loop until we get back to the start
        loop.append(next_point)

        # Find the points connected to the current point (called next from previous iteration) and then find the
        # one that is not the previous point (where came from) as that is where we are going.
        points_connected_to_next = [pipe_network[y][x] for x, y in next_point.positions_connected_to()]
        points_connected_back_to_next = [point for point in points_connected_to_next if next_point.coordinates() in
                                         point.positions_connected_to()]
        potential_next_points = [point for point in points_connected_back_to_next if point != previous_point]

        # Should be unique, if not something has gone wrong
        if len(potential_next_points) == 1:
            previous_point = next_point
            next_point = potential_next_points[0]
        else:
            raise ValueError(f'Unexpected number of possible next points in loop {len(potential_next_points)=}')

    return loop


# Enum to store inside/outside status of a tile for part 2
class TileStatus(Enum):
    INSIDE = auto()
    OUTSIDE = auto()
    INDETERMINATE_ENTER_SE = auto()  # Can be indeterminate, for example, if we enter from above F
    INDETERMINATE_ENTER_SW = auto()  # we don't know if we are now inside or outside, it depends on if we encounter
                                     # a L or J next


def opposite_tile_status(tile_status: TileStatus) -> TileStatus:
    """Find the opposite tile status."""
    if tile_status == TileStatus.INSIDE:
        return TileStatus.OUTSIDE
    elif tile_status == TileStatus.OUTSIDE:
        return TileStatus.INSIDE
    else:
        raise ValueError(f'Opposite tile status not defined for {tile_status=}')


def find_enclosed_tiles(pipe_loop: List[PipePoint], pipe_network: List[List[PipePoint]]) -> List[Tuple[int, int]]:
    """Find the enclosed tile coordinates in the pipe network."""
    logging.log(logging.DEBUG, f'Finding enclosed tiles.')

    # Find the minimum and maximum x and y coordinates of pipe loop as these are the bounds that define possible
    # enclosed tiles
    pipe_loop_coordinates = [point.coordinates() for point in pipe_loop]
    min_x = min([x for x, y in pipe_loop_coordinates])
    max_x = max([x for x, y in pipe_loop_coordinates])
    min_y = min([y for x, y in pipe_loop_coordinates])
    max_y = max([y for x, y in pipe_loop_coordinates])

    # Inside/outside can be defined by the parity of the number of pipe loop crossings to get to the border of the grid.
    # e.g. 0 is outside, 1 is inside, 2 is outside, 3 is inside, 4 is outside, etc.
    # Naively, we could just exit from every tile in an arbitrary direction and count the number of times we cross the
    # pipe loop. However, this is inefficient as we will be counting the same crossings multiple times.
    # Instead, we can consider crossing the loop from top to bottom (in each column), keeping track of the parity and
    # then using that to assign to cells if they are inside or outside as we go along
    enclosed_tiles = []
    for x in range(min_x + 1, max_x):  # Left and right columns are always outside by construction on a square grid
        logging.log(logging.DEBUG, f'Finding enclosed tiles in column {x}.')

        # Start at the top of the column moving downwards
        current_tile_status = TileStatus.OUTSIDE
        previous_tile_status = TileStatus.OUTSIDE  # Used when indeterminate
        for y in range(min_y, max_y + 1):
            pipe_point = pipe_network[y][x]

            # If the pipe point is not inside the loop, our status will not be changed.
            # We just need to record the tile if we are inside.
            if pipe_point not in pipe_loop:
                if current_tile_status == TileStatus.INSIDE:
                    enclosed_tiles.append(pipe_point.coordinates())
                continue

            # If the pipe point is inside the loop, we may need to update our status.
            # This is quite ugly, there is probably a nicer way to code this.
            if current_tile_status == TileStatus.OUTSIDE:  # Was outside so now inside or indeterminate
                if pipe_point.type == horizontal_pipe:
                    current_tile_status = TileStatus.INSIDE
                elif pipe_point.type == se_pipe:
                    current_tile_status = TileStatus.INDETERMINATE_ENTER_SE
                    previous_tile_status = TileStatus.OUTSIDE
                elif pipe_point.type == sw_pipe:
                    current_tile_status = TileStatus.INDETERMINATE_ENTER_SW
                    previous_tile_status = TileStatus.OUTSIDE
                else:
                    raise ValueError(f'Unexpected pipe type on entering pipe loop {pipe_point.type=}')

            elif current_tile_status == TileStatus.INSIDE:  # Inside so now outside or indeterminate
                if pipe_point.type == horizontal_pipe:
                    current_tile_status = TileStatus.OUTSIDE
                elif pipe_point.type == se_pipe:
                    current_tile_status = TileStatus.INDETERMINATE_ENTER_SE
                    previous_tile_status = TileStatus.INSIDE
                elif pipe_point.type == sw_pipe:
                    current_tile_status = TileStatus.INDETERMINATE_ENTER_SW
                    previous_tile_status = TileStatus.INSIDE
                else:
                    raise ValueError(f'Unexpected pipe type on exiting pipe loop {pipe_point.type=}')

            elif current_tile_status == TileStatus.INDETERMINATE_ENTER_SE:
                if pipe_point.type == vertical_pipe:
                    # No new information, leave status unchanged
                    continue
                elif pipe_point.type == ne_pipe: # Resolve indeterminate status
                    current_tile_status = previous_tile_status
                elif pipe_point.type == nw_pipe:
                    current_tile_status = opposite_tile_status(previous_tile_status)
                else:
                    raise ValueError(f'Unexpected pipe type when inside pipe loop {pipe_point.type=}')

            elif current_tile_status == TileStatus.INDETERMINATE_ENTER_SW:
                if pipe_point.type == vertical_pipe:
                    # No new information, leave status unchanged
                    continue
                elif pipe_point.type == ne_pipe: # Resolve indeterminate status
                    current_tile_status = opposite_tile_status(previous_tile_status)
                elif pipe_point.type == nw_pipe:
                    current_tile_status = previous_tile_status
                else:
                    raise ValueError(f'Unexpected pipe type when inside pipe loop {pipe_point.type=}')

            else:
                raise ValueError(f'Unexpected tile status {current_tile_status=}')


    return enclosed_tiles


def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    pipe_network = load_pipe_network()
    logging.log(logging.DEBUG, f'{pipe_network=}')
    logging.log(logging.DEBUG, f'{pipe_network[69][88].type}')

    # Find the pipe loop
    pipe_loop = find_pipe_loop(pipe_network)

    # The required answer is the furthest step distance from the start of the pipe in either direction.
    # Setup on a square grid means this distance is always half the length of the pipe loop (which is always even).
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 10 part 1, is: {int(len(pipe_loop)/2)}')

    # Now we have the loop replace the start pipe as no longer special with what it should be
    start_pipe = pipe_loop[0]
    start_pipe.type = se_pipe  # Found by inspection as much simpler than coding all the cases

    # Find the number of tiles enclosed by the pipe loop
    enclosed_tiles = find_enclosed_tiles(pipe_loop, pipe_network)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 10 part 2, is: {len(enclosed_tiles)}')



if __name__ == '__main__':
    main()

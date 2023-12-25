"""Day Sixteen of Advent of Code 2023."""

# Import required packages
import logging
import numpy as np
from typing import List
from dataclasses import dataclass
from enum import Enum, auto


# Parameters
input_file = 'input.txt'


# Data structures
class Direction(Enum):
    """An enum for the directions."""
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()

    @classmethod
    def opposite(cls, direction: 'Direction') -> 'Direction':
        """Return the opposite direction."""
        if direction == cls.UP:
            return cls.DOWN
        elif direction == cls.RIGHT:
            return cls.LEFT
        elif direction == cls.DOWN:
            return cls.UP
        elif direction == cls.LEFT:
            return cls.RIGHT
        else:
            raise ValueError(f'Unknown direction {direction}.')


@dataclass
class Ray:
    """A ray of light in the machine."""
    x: int
    y: int
    direction: Direction

    def set_position(self, x: int, y: int) -> 'Ray':
        """Set the position of the ray."""
        self.x = x
        self.y = y
        return self


@dataclass
class MachineTile:
    """A tile in the machine."""
    x: int
    y: int
    top_active: bool
    bottom_active: bool
    left_active: bool
    right_active: bool

    def activate(self, direction: Direction) -> None:
        """Activate the tile in the given direction."""
        if direction == Direction.UP:
            self.top_active = True
        elif direction == Direction.RIGHT:
            self.right_active = True
        elif direction == Direction.DOWN:
            self.bottom_active = True
        elif direction == Direction.LEFT:
            self.left_active = True
        else:
            raise ValueError(f'Unknown direction {direction}.')


    def is_active(self, direction: Direction) -> bool:
        """Return whether the tile is active in the given direction."""
        if direction == Direction.UP:
            return self.top_active
        elif direction == Direction.RIGHT:
            return self.right_active
        elif direction == Direction.DOWN:
            return self.bottom_active
        elif direction == Direction.LEFT:
            return self.left_active
        else:
            raise ValueError(f'Unknown direction {direction}.')


    def is_active_any(self) -> bool:
        """Return whether the tile is active in any direction."""
        return self.top_active or self.right_active or self.bottom_active or self.left_active


    def transform_entering_ray(self, ray: Ray) -> List[Ray]:
        """Transform ray entering the tile."""
        # Does nothing by default just activates the tile
        self.activate(Direction.opposite(ray.direction))
        return [ray.set_position(self.x, self.y)]


# Define the types of machine tile
class EmptyTile(MachineTile):
    """An empty tile in the machine."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, False, False, False, False)  # All sides inactive to start


class RisingMirrorTile(MachineTile):  # /
    """A rising mirror tile in the machine, e.g. /."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, False, False, False, False)

    def transform_entering_ray(self, ray: Ray) -> List[Ray]:
        """Transform ray entering the tile."""
        self.activate(Direction.opposite(ray.direction))

        # Determine the new direction of the ray
        if ray.direction == Direction.UP:
            new_direction = Direction.RIGHT
        elif ray.direction == Direction.RIGHT:
            new_direction = Direction.UP
        elif ray.direction == Direction.DOWN:
            new_direction = Direction.LEFT
        elif ray.direction == Direction.LEFT:
            new_direction = Direction.DOWN
        else:
            raise ValueError(f'Unknown direction {ray.direction}.')

        # Update ray position and direction
        ray.set_position(self.x, self.y)
        ray.direction = new_direction
        return [ray]


class FallingMirrorTile(MachineTile):  # \
    """A falling mirror tile in the machine, e.g. \."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, False, False, False, False)

    def transform_entering_ray(self, ray: Ray) -> List[Ray]:
        """Transform ray entering the tile."""
        self.activate(Direction.opposite(ray.direction))

        # Determine the new direction of the ray
        if ray.direction == Direction.UP:
            new_direction = Direction.LEFT
        elif ray.direction == Direction.RIGHT:
            new_direction = Direction.DOWN
        elif ray.direction == Direction.DOWN:
            new_direction = Direction.RIGHT
        elif ray.direction == Direction.LEFT:
            new_direction = Direction.UP
        else:
            raise ValueError(f'Unknown direction {ray.direction}.')

        # Update ray position and direction
        ray.set_position(self.x, self.y)
        ray.direction = new_direction
        return [ray]


class HorizontalSplitterTile(MachineTile):  # -
    """A horizontal splitter tile in the machine, e.g. -."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, False, False, False, False)

    def transform_entering_ray(self, ray: Ray) -> List[Ray]:
        """Transform ray entering the tile."""
        self.activate(Direction.opposite(ray.direction))

        # Left-right splitter, so does nothing if enters from left or right
        if ray.direction in [Direction.LEFT, Direction.RIGHT]:
            return [ray.set_position(self.x, self.y)]

        # Otherwise, the ray splits into two rays as it is entering from above or below
        elif ray.direction in [Direction.UP, Direction.DOWN]:
            return [Ray(self.x, self.y, Direction.LEFT), Ray(self.x, self.y, Direction.RIGHT)]

        else:
            raise ValueError(f'Unknown direction {ray.direction}.')


class VerticalSplitterTile(MachineTile):  # |
    """A vertical splitter tile in the machine, e.g. |."""
    def __init__(self, x: int, y: int):
        super().__init__(x, y, False, False, False, False)

    def transform_entering_ray(self, ray: Ray) -> List[Ray]:
        """Transform ray entering the tile."""
        self.activate(Direction.opposite(ray.direction))

        # Up-down splitter, so does nothing if enters from above or below
        if ray.direction in [Direction.UP, Direction.DOWN]:
            return [ray.set_position(self.x, self.y)]

        # Otherwise, the ray splits into two rays as it is entering from left or right
        elif ray.direction in [Direction.LEFT, Direction.RIGHT]:
            return [Ray(self.x, self.y, Direction.UP), Ray(self.x, self.y, Direction.DOWN)]

        else:
            raise ValueError(f'Unknown direction {ray.direction}.')


# Define mapping from symbols to machine tiles for reading in the input
symbol_to_tile = {
    '.': EmptyTile,
    '/': RisingMirrorTile,
    '\\': FallingMirrorTile,
    '-': HorizontalSplitterTile,
    '|': VerticalSplitterTile
}


# IO
def load_tile_map() -> np.ndarray:
    """Load the tile map from the input file."""
    logging.log(logging.DEBUG, f'Loading tile map from {input_file}.')

    # Get array of tile types
    with open(input_file, 'r') as f:
        tile_map = [[symbol_to_tile[symbol] for symbol in line.strip()] for line in f.readlines()]
    array = np.array(tile_map)

    # Initialize the tile map
    for y_idx in range(array.shape[0]):
        for x_idx in range(array.shape[1]):
            array[y_idx, x_idx] = array[y_idx, x_idx](x_idx, y_idx)

    return array



# Propagation of rays
def propagate_ray(ray: Ray, tile_map: np.ndarray) -> List[Ray]:
    """Propagate the given ray forward one step."""
    # Find the next position
    ray_x, ray_y = ray.x, ray.y
    if ray.direction == Direction.UP:
        ray_y -= 1
    elif ray.direction == Direction.RIGHT:
        ray_x += 1
    elif ray.direction == Direction.DOWN:
        ray_y += 1
    elif ray.direction == Direction.LEFT:
        ray_x -= 1
    else:
        raise ValueError(f'Unknown direction {ray.direction}.')

    # Check if the ray has left the machine
    if ray_x < 0 or ray_x >= tile_map.shape[1] or ray_y < 0 or ray_y >= tile_map.shape[0]:
        return []

    # Find tile being entered
    tile = tile_map[ray_y, ray_x]

    # If tile has already been entered, in this direction, then in loop so stop
    if tile.is_active(Direction.opposite(ray.direction)):
        return []

    # Otherwise, transform the ray and propagate it
    return tile.transform_entering_ray(ray)

def illuminate_machine(tile_map: np.ndarray, initial_ray: Ray = None) -> None:
    """Illuminate the machine"""
    logging.log(logging.DEBUG, f'Illuminating the machine.')

    # Setup initial ray
    if initial_ray is None:
        initial_ray = Ray(-1, 0, Direction.RIGHT)
    rays = [initial_ray]

    # Loop until no more rays
    while len(rays) > 0:
        current_ray = rays.pop(0)
        rays += propagate_ray(current_ray, tile_map)


def find_maximum_num_active(tile_map: np.ndarray) -> int:
    """Find the maximum number of activated tiles for any valid initial ray."""
    logging.log(logging.DEBUG, f'Finding the maximum number of activated tiles for any valid initial ray.')

    # Find all initial rays
    tile_map_shape = tile_map.shape
    rightward_rays = [Ray(-1, y_idx, Direction.RIGHT) for y_idx in range(tile_map_shape[0])]
    leftward_rays = [Ray(tile_map_shape[1], y_idx, Direction.LEFT) for y_idx in range(tile_map_shape[0])]
    downward_rays = [Ray(x_idx, -1, Direction.DOWN) for x_idx in range(tile_map_shape[1])]
    upward_rays = [Ray(x_idx, tile_map_shape[0], Direction.UP) for x_idx in range(tile_map_shape[1])]
    initial_rays = rightward_rays + leftward_rays + downward_rays + upward_rays

    # Loop through initial rays, illuminating the machine and finding the maximum number of activated tiles
    maximum_num_active = 0
    for initial_ray in initial_rays:
        # Deactivate all tiles
        for tile in tile_map.flatten():
            tile.top_active = False
            tile.right_active = False
            tile.bottom_active = False
            tile.left_active = False

        illuminate_machine(tile_map, initial_ray)
        num_active = sum([tile.is_active_any() for tile in tile_map.flatten()])
        maximum_num_active = max(num_active, maximum_num_active)

    return maximum_num_active


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    tile_map = load_tile_map()

    # Part I, Find the number of activated tiles
    illuminate_machine(tile_map)
    num_active = sum([tile.is_active_any() for tile in tile_map.flatten()])
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 16 part 1, is: '
                              f'{num_active}')

    # Part II, Find the maximum number of activated tiles for any valid initial ray
    maximum_num_active = find_maximum_num_active(tile_map)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 16 part 2, is: '
                              f'{maximum_num_active}')


if __name__ == '__main__':
    main()

"""Day Twenty Four of Advent of Code 2023."""

# Import required packages
import logging
from typing import List, Tuple
import numpy as np
import fractions
from dataclasses import dataclass
from math import gcd


# Parameters
input_file = 'input.txt'


# Constants
position_velocity_spliter = '@'
coordinate_delimiter = ','

example_min_coord = 7
example_max_coord = 27
true_min_coord = 200000000000000
true_max_coord = 400000000000000


# Data structures
@dataclass
class HailstoneRecord:
    """Data structure to store a hailstone record."""
    initial_x: int
    initial_y: int
    initial_z: int
    velocity_x: int
    velocity_y: int
    velocity_z: int

    @property
    def xy_line_gradient(self) -> fractions.Fraction:
        """Calculate the gradient of the projected trajectory on the xy plane."""
        return fractions.Fraction(self.velocity_y, self.velocity_x)

    @property
    def xy_line_intercept(self) -> fractions.Fraction:
        """Calculate the intercept of the projected trajectory on the xy plane."""
        return fractions.Fraction(self.initial_y * self.velocity_x - self.initial_x * self.velocity_y,
                                  self.velocity_x)

    @property
    def xz_line_gradient(self) -> fractions.Fraction:
        """Calculate the gradient of the projected trajectory on the xz plane."""
        return fractions.Fraction(self.velocity_z, self.velocity_x)

    @property
    def xz_line_intercept(self) -> fractions.Fraction:
        """Calculate the intercept of the projected trajectory on the xz plane."""
        return fractions.Fraction(self.initial_z * self.velocity_x - self.initial_x * self.velocity_z,
                                  self.velocity_x)

    @property
    def yz_line_gradient(self) -> fractions.Fraction:
        """Calculate the gradient of the projected trajectory on the yz plane."""
        return fractions.Fraction(self.velocity_z, self.velocity_y)

    @property
    def yz_line_intercept(self) -> fractions.Fraction:
        """Calculate the intercept of the projected trajectory on the yz plane."""
        return fractions.Fraction(self.initial_z * self.velocity_y - self.initial_y * self.velocity_z,
                                  self.velocity_y)

    def position_at_time(self, time: int) -> (int, int, int):
        """Calculate the position of the hailstone at a given time."""
        x = self.initial_x + self.velocity_x * time
        y = self.initial_y + self.velocity_y * time
        z = self.initial_z + self.velocity_z * time
        return x, y, z

    def time_at_position(self, position: (int, int, int)) -> int:
        """Calculate the time at which the hailstone is at a given position."""
        x, y, z = position

        # Calculate time in each dimension
        times = []
        if self.velocity_x != 0:
            time_x = int((x - self.initial_x) / self.velocity_x)
            times.append(time_x)
        if self.velocity_y != 0:
            time_y = int((y - self.initial_y) / self.velocity_y)
            times.append(time_y)
        if self.velocity_z != 0:
            time_z = int((z - self.initial_z) / self.velocity_z)
            times.append(time_z)

        # Check if the times are the same
        if len(set(times)) != 1:
            raise ValueError(f'Position {position} is not on the hailstone trajectory.')

        # Return the time
        time = times[0]
        return time


# IO
def load_input(file_path: str) -> List[HailstoneRecord]:
    """Load the input file and convert to a list of hailstone records."""
    with open(file_path, 'r') as f:
        records = []
        for line in f.readlines():
            position, velocity = line.strip().split(position_velocity_spliter)
            x, y, z = position.strip().split(coordinate_delimiter)
            x, y, z = int(x), int(y), int(z)
            vx, vy, vz = velocity.strip().split(coordinate_delimiter)
            vx, vy, vz = int(vx), int(vy), int(vz)
            records.append(HailstoneRecord(x, y, z, vx, vy, vz))

        return records


# Data processing
def calculate_num_intersections(hailstone_records: List[HailstoneRecord],
                                min_coord: int,
                                max_coord: int,
                                projection: str = 'z',
                                require_future: bool = True) -> int:
    """Calculate the number of intersections in the test area."""
    logging.log(logging.DEBUG, f'Calculating the number of intersections in the test area')

    # Calculate the number of intersections by pairwise comparison
    num_intersections = 0
    for idx, hailstone_1 in enumerate(hailstone_records):
        for hailstone_2 in hailstone_records[idx + 1:]:
            # In 2D space, the lines are guaranteed to intersect if the gradients are different
            if projection == 'z':
                gradient_1 = hailstone_1.xy_line_gradient
                gradient_2 = hailstone_2.xy_line_gradient
            elif projection == 'y':
                gradient_1 = hailstone_1.xz_line_gradient
                gradient_2 = hailstone_2.xz_line_gradient
            elif projection == 'x':
                gradient_1 = hailstone_1.yz_line_gradient
                gradient_2 = hailstone_2.yz_line_gradient
            else:
                raise ValueError(f'Unknown projection: {projection}')

            if gradient_1 == gradient_2:
                continue

            # Calculate the intersection point
            if projection == 'z':
                intercept_1 = hailstone_1.xy_line_intercept
                intercept_2 = hailstone_2.xy_line_intercept
            elif projection == 'y':
                intercept_1 = hailstone_1.xz_line_intercept
                intercept_2 = hailstone_2.xz_line_intercept
            elif projection == 'x':
                intercept_1 = hailstone_1.yz_line_intercept
                intercept_2 = hailstone_2.yz_line_intercept
            else:
                raise ValueError(f'Unknown projection: {projection}')
            axis1_intersect = (intercept_2 - intercept_1) / (gradient_1 - gradient_2)
            axis2_intersect = (gradient_1 * intercept_2 - gradient_2 * intercept_1) / (gradient_1 - gradient_2)

            # Check if the intersection point is in the future
            if require_future:
                if projection == 'x':
                    if (axis1_intersect - hailstone_1.initial_y) / hailstone_1.velocity_y < 0:
                        continue
                    if (axis1_intersect - hailstone_2.initial_y) / hailstone_2.velocity_y < 0:
                        continue
                elif projection == 'y' or projection == 'z':
                    if (axis1_intersect - hailstone_1.initial_x) / hailstone_1.velocity_x < 0:
                        continue
                    if (axis1_intersect - hailstone_2.initial_x) / hailstone_2.velocity_x < 0:
                        continue

            # Check if the intersection point is in the test area
            if min_coord <= axis1_intersect <= max_coord and min_coord <= axis2_intersect <= max_coord:
                num_intersections += 1

    return num_intersections


def find_intersection_point(hailstone: HailstoneRecord, intersection_vector: (int, int, int)) -> (int, int, int):
    """Find the intersection point of a hailstone and a line through the origin."""
    logging.log(logging.DEBUG, f'Finding the intersection point of a hailstone and a line')

    # Find the intersection time
    intersection_time = fractions.Fraction(
        intersection_vector[0] * hailstone.initial_y - intersection_vector[1] * hailstone.initial_x,
        intersection_vector[1] * hailstone.velocity_x - intersection_vector[0] * hailstone.velocity_y)
    intersection_time_z = fractions.Fraction(
        intersection_vector[0] * hailstone.initial_z - intersection_vector[2] * hailstone.initial_x,
        intersection_vector[2] * hailstone.velocity_x - intersection_vector[0] * hailstone.velocity_z)

    if intersection_time != intersection_time_z:
        raise ValueError(f'Do not intersect.')

    # Find the intersection point
    intersection_point = (hailstone.initial_x + hailstone.velocity_x * intersection_time,
                          hailstone.initial_y + hailstone.velocity_y * intersection_time,
                          hailstone.initial_z + hailstone.velocity_z * intersection_time)
    return intersection_point


def cross_product(a: (int, int, int), b: (int, int, int)) -> (int, int, int):
    """Calculate the cross-product of two vectors. Need to write our own for arbitrary precision."""
    return a[1] * b[2] - a[2] * b[1], \
           a[2] * b[0] - a[0] * b[2], \
           a[0] * b[1] - a[1] * b[0]


def where_to_stand(hailstone_records: List[HailstoneRecord]) -> Tuple[int, ...]:
    """Find where to stand for the perfect throw.

    We shall exploit the Galilean transformation to find the perfect throw.
    """
    logging.log(logging.DEBUG, f'Finding where to stand for the perfect throw')

    # Transform into the rest frame of the first hailstone
    first_hailstone = hailstone_records[0]
    transformed_hailstone_records = []
    for hailstone in hailstone_records:
        transformed_hailstone_records.append(HailstoneRecord(hailstone.initial_x - first_hailstone.initial_x,
                                                             hailstone.initial_y - first_hailstone.initial_y,
                                                             hailstone.initial_z - first_hailstone.initial_z,
                                                             hailstone.velocity_x - first_hailstone.velocity_x,
                                                             hailstone.velocity_y - first_hailstone.velocity_y,
                                                             hailstone.velocity_z - first_hailstone.velocity_z))

    # Each other hailstone is now moving on a straight line in this frame, while the first hailstone is stationary
    # at the origin. Hence, any throw that hits the first hailstone and any other has to have a velocity in the
    # plane defined by the origin and the other hailstone's trajectory line. Two such planes, then, define a line
    # along which the throw must be made.

    plane_normal_vectors = []
    for hailstone_to_hit in transformed_hailstone_records[1:3]:
        # Find the normal vector of the plane defined by the origin and the other hailstone's trajectory line.
        # Since the point is the origin I have dropped some +- zero terms.
        velocity_of_moving = (hailstone_to_hit.velocity_x, hailstone_to_hit.velocity_y, hailstone_to_hit.velocity_z)
        origin_of_moving = (hailstone_to_hit.initial_x, hailstone_to_hit.initial_y, hailstone_to_hit.initial_z)
        normal_vector = cross_product(velocity_of_moving, origin_of_moving)
        plane_normal_vectors.append(normal_vector)

    # Find the intersection of the two planes (none-normalized throw velocity)
    intersection_vector = cross_product(plane_normal_vectors[0], plane_normal_vectors[1])
    simplification_factor = gcd(intersection_vector[0], gcd(intersection_vector[1], intersection_vector[2]))
    intersection_vector = [iv / simplification_factor for iv in intersection_vector]
    intersection_vector = tuple([int(i) for i in intersection_vector])

    # Normalize the throw velocity using intersection times
    intersection_1 = find_intersection_point(transformed_hailstone_records[1], intersection_vector)
    intersection_2 = find_intersection_point(transformed_hailstone_records[2], intersection_vector)
    intersection_1 = tuple([int(i) for i in intersection_1])
    intersection_2 = tuple([int(i) for i in intersection_2])
    d_1 = np.linalg.norm(intersection_1)
    d_2 = np.linalg.norm(intersection_2)
    t_1 = transformed_hailstone_records[1].time_at_position(intersection_1)
    t_2 = transformed_hailstone_records[2].time_at_position(intersection_2)
    throw_velocity = np.array(intersection_vector) * ((d_2 - d_1)/(t_2 - t_1))/ np.linalg.norm(intersection_vector)

    # Transform back into the original frame
    throw_velocity = throw_velocity + \
                     (first_hailstone.velocity_x, first_hailstone.velocity_y, first_hailstone.velocity_z)
    throw_velocity = tuple([int(i) for i in throw_velocity])

    # Get the position of a hailstone at known time and convert to throw position
    hit_position = hailstone_records[1].position_at_time(t_1)
    throw_position = np.array(hit_position) - np.array(throw_velocity) * t_1
    throw_position = tuple([int(i) for i in throw_position])
    return throw_position


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    hailstone_records = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded {len(hailstone_records)} hailstone records')

    # Get test area information
    if input_file == 'example.txt':
        min_coord = example_min_coord
        max_coord = example_max_coord
    else:
        min_coord = true_min_coord
        max_coord = true_max_coord

    # Part I, find the number of stones paths that intersect in the test area
    num_intersections = calculate_num_intersections(hailstone_records, min_coord, max_coord)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 24 part 1, is: '
                                f'{num_intersections}')

    # Part II, find where to stand for the perfect throw. Note in general, such a throw is not possible.
    # For this solution, I am going to assume it is.
    x_throw, y_throw, z_throw = where_to_stand(hailstone_records)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 24 part 2, is: '
                                f'{x_throw + y_throw + z_throw}')


if __name__ == '__main__':
    main()

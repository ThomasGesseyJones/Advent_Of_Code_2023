"""Day Nineteen of Advent of Code 2023."""

# Import required packages
from __future__ import annotations
import logging
from typing import List, Tuple


# Parameters
input_file = 'input.txt'


# Constants
start_workflow = 'in'

workflow_begin = '{'
workflow_end = '}'
command_delimiter = ','
command_key_value_delimiter = ':'
gt_symbol = '>'
lt_symbol = '<'
accept_symbol = 'A'
reject_symbol = 'R'

part_info_begin = '{'
part_info_end = '}'
part_info_delimiter = ','
part_info_key_value_delimiter = '='

min_part_score = 1  # inclusive
max_part_score = 4000  # inclusive


# IO
def load_input(file_path: str) -> Tuple[List[str], List[str]]:
    """Load the input file."""
    logging.log(logging.DEBUG, f'Loading input file: {file_path}')

    # Data structures to store the input
    workflow_strs = []
    part_info_strs = []

    # First half of the file is the workflows, then a blank line, then the part info
    with open(file_path, 'r') as f:
        reading_workflows = True
        for line in f:
            line = line.strip()

            if line == '':
                reading_workflows = False
                continue

            if reading_workflows:
                workflow_strs.append(line)
            else:
                part_info_strs.append(line)

    return workflow_strs, part_info_strs


# Parsing
def parse_part_info(part_info_strs: List[str]) -> List[dict]:
    """Parse the part info strings."""
    logging.log(logging.DEBUG, f'Parsing {len(part_info_strs)} part info strings.')

    # Will store each piece of part info as a dictionary for O(1) lookup of its properties
    part_info_list = []
    for part_info_str in part_info_strs:
        part_info = {}

        # Remove the surrounding brackets
        part_info_str = part_info_str.replace(part_info_begin, '').replace(part_info_end, '')

        # Split the part info string into key-value pairs (category-score)
        for part_info_key_value_str in part_info_str.split(part_info_delimiter):
            part_info_key, part_info_value = part_info_key_value_str.split(part_info_key_value_delimiter)
            part_info[part_info_key] = int(part_info_value)  # Scores seem to be ints
        part_info_list.append(part_info)

    return part_info_list


def parse_workflow(workflow_strs: List[str]) -> dict:
    """Parse the workflow strings into functions using a recursive functional approach.

    A workflow command is a function that takes a part range dictionary and returns
    the number of parts that are accepted by the workflow within that range. The workflows
    can call one another using a dictionary lookup.
    """
    logging.log(logging.DEBUG, f'Parsing {len(workflow_strs)} workflow strings.')

    # Each workflow will be a function, which we will store in a dictionary labeled by its name.
    # By lookup-ing within the dictionary, functions will be able to call one another.
    workflow_dict = {}

    # Recursive function converter
    def workflow_to_function(workflow: List[str]) -> callable:
        """Convert a workflow to a function."""
        # Should never be empty, but just in case
        if len(workflow) == 0:
            raise ValueError('Workflow is empty.')

        # Get current command from start of workflow
        command = workflow.pop(0)

        # If end state
        if command == accept_symbol:
            # Accept all parts within the range
            def accept(x: dict) -> int:
                num_accepted = 1
                for v_min, v_max in x.values():
                    num_accepted *= (v_max - v_min + 1)
                return num_accepted
            return accept

        elif command == reject_symbol:
            # Reject all parts within the range
            return lambda x: 0

        # If none conditional
        if command_key_value_delimiter not in command:
            # Pass the range to a different workflow
            return lambda x: workflow_dict[command](x)

        # Otherwise conditional
        command_key, command_value = command.split(command_key_value_delimiter)

        # Need to consider the different possible conditional operators and how they affect the range
        if gt_symbol in command_key:
            field, value = command_key.split(gt_symbol)  # field is the part info category, value is the score

            # What range is accepted by the condition?
            def get_true_range(part_range: dict) -> dict | None:
                """Get the part range that satisfies the condition."""
                accepted_range = part_range.copy()
                accepted_range[field] = (max(int(value) + 1, accepted_range[field][0]), accepted_range[field][1])
                if accepted_range[field][0] > accepted_range[field][1]:
                    return None  # No parts can satisfy this condition
                return accepted_range

            # What range is rejected by the condition?
            def get_false_range(part_range: dict) -> dict | None:
                """Get the part range that does not satisfy the condition."""
                rejected_range = part_range.copy()
                rejected_range[field] = (rejected_range[field][0], min(int(value), rejected_range[field][1]))
                if rejected_range[field][0] > rejected_range[field][1]:
                    return None  # All parts satisfy this condition
                return rejected_range

        elif lt_symbol in command_key:
            field, value = command_key.split(lt_symbol)

            # What range is accepted by the condition?
            def get_true_range(part_range: dict) -> dict | None:
                """Get the part range that satisfies the condition."""
                accepted_range = part_range.copy()
                accepted_range[field] = (accepted_range[field][0], min(int(value) - 1, accepted_range[field][1]))
                if accepted_range[field][0] > accepted_range[field][1]:
                    return None
                return accepted_range

            # What range is rejected by the condition?
            def get_false_range(part_range: dict) -> dict | None:
                """Get the part range that does not satisfy the condition."""
                rejected_range = part_range.copy()
                rejected_range[field] = (max(int(value), rejected_range[field][0]), rejected_range[field][1])
                if rejected_range[field][0] > rejected_range[field][1]:
                    return None
                return rejected_range

        else:
            raise ValueError(f'Unknown command key: {command_key}')

        # Workflow splits at a condition, so we need to run the true and false workflows separately on the accepted and
        # rejected ranges
        fn_true = workflow_to_function([command_value])
        fn_false = workflow_to_function(workflow)

        def conditional_workflow(part_range: dict) -> int:
            """Run a conditional workflow to find the number of accepted parts."""
            num_accepted = 0
            true_range = get_true_range(part_range)
            false_range = get_false_range(part_range)
            if true_range is not None:
                num_accepted += fn_true(true_range)
            if false_range is not None:
                num_accepted += fn_false(false_range)
            return num_accepted

        return conditional_workflow

    # Parse each workflow string using the recursive function converter
    for workflow_str in workflow_strs:
        workflow_lbl, workflow_str = workflow_str.split(workflow_begin)
        workflow_str = workflow_str.replace(workflow_end, '')
        logging.log(logging.DEBUG, workflow_str)
        workflow_dict[workflow_lbl] = workflow_to_function(workflow_str.split(command_delimiter))

    return workflow_dict


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    workflow_strs, part_info_strs = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded {len(workflow_strs)} workflow strings and '
                               f'{len(part_info_strs)} part info strings.')

    # Parse the input
    workflow_dict = parse_workflow(workflow_strs)
    part_info_list = parse_part_info(part_info_strs)

    # Part I, find the accepted parts and sum their scores
    total_score = 0
    for part_info in part_info_list:
        logging.log(logging.DEBUG, f'Running workflow on part info: {part_info}')

        # Convert the part info to required format of a part range (changed for part II)
        part_range = {k: (v, v) for k, v in part_info.items()}

        # Run the workflow on the part range to see if it is accepted (should give 1 if accepted, 0 if not)
        accepted = workflow_dict[start_workflow](part_range)
        if accepted:
            total_score += int(sum(part_info.values()))

    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 19 part 1, is: '
                              f'{total_score}')

    # Part II, find how many parts could be accepted
    allowed_ranges = {'x': (min_part_score, max_part_score),
                      'm': (min_part_score, max_part_score),
                      'a': (min_part_score, max_part_score),
                      's': (min_part_score, max_part_score)}
    num_accepted = workflow_dict[start_workflow](allowed_ranges)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 19 part 2, is: '
                              f'{num_accepted}')


if __name__ == '__main__':
    main()

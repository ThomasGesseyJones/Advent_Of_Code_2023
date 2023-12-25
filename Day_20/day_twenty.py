"""Day Twenty of Advent of Code 2023."""

# Import required packages
from __future__ import annotations
import logging
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from math import gcd


# Parameters
input_file = 'input.txt'


# Constants
flip_flop_symbol = '%'
conjunct_symbol = '&'
output_delimiter = ','
input_output_split_symbol = '->'


# Data structure to represent the machine
class PulseStrength(Enum):
    """Enum to represent the strength of a pulse."""
    LOW = auto()
    HIGH = auto()


@dataclass
class MachineModule(ABC):
    """Base class for types of modules in the machine."""
    name: str
    type: str
    input: Tuple[MachineModule, ...]
    output: Tuple[MachineModule, ...]
    state: Tuple

    def __eq__(self, other: MachineModule) -> bool:
        """Compare two modules. They are equal if all fields are equal."""
        return (self.name == other.name and
                self.type == other.type and
                self.input == other.input and
                self.output == other.output and
                self.state == other.state)

    def add_input(self, input_module: MachineModule) -> None:
        """Add an input module."""
        self.input += (input_module,)

    def add_output(self, output_module: MachineModule) -> None:
        """Add an output module."""
        self.output += (output_module,)

    @abstractmethod
    def process_pulse(self, from_input: MachineModule,
                      pulse_strength: PulseStrength) -> List[Tuple[MachineModule, PulseStrength]]:
        """Process a pulse of given strength."""
        pass

    def __str__(self):
        return f'{self.name} ({self.type})'

    def __repr__(self):
        return str(self)


class ButtonModule(MachineModule):
    """Class to represent the button module (the start of the machine)."""
    def __init__(self) -> None:
        """Initialise the button module."""
        logging.log(logging.DEBUG, f'Initialising button module')
        super().__init__('button', 'button', tuple(), tuple(), tuple())

    def process_pulse(self, from_input: MachineModule,
                      pulse_strength: PulseStrength) -> List[Tuple[MachineModule, PulseStrength]]:
        """Process a pulse of given strength."""
        # If the pulse is high, then we can send a pulse to the output
        if pulse_strength == PulseStrength.HIGH:
            return [(output, PulseStrength.LOW) for output in self.output]
        else:
            raise ValueError(f'Button module should never receive a pulse of strength {pulse_strength}')


class BroadcasterModule(MachineModule):
    """Class to represent the broadcaster module."""
    def __init__(self) -> None:
        """Initialise the broadcaster module."""
        logging.log(logging.DEBUG, f'Initialising broadcaster module')
        super().__init__('broadcaster', 'broadcaster', tuple(), tuple(), tuple())


    def process_pulse(self, from_input: MachineModule,
                      pulse_strength: PulseStrength) -> List[Tuple[MachineModule, PulseStrength]]:
        """Process a pulse of given strength."""
        # The Broadcaster sends the pulse to all outputs
        return [(output, pulse_strength) for output in self.output]


class FlipFlopModule(MachineModule):
    """Class to represent a flip-flop module."""
    def __init__(self, name: str) -> None:
        """Initialise the flip-flop module."""
        logging.log(logging.DEBUG, f'Initialising flip-flop module {name}')
        super().__init__(name, flip_flop_symbol, tuple(), tuple(), (False,))


    def process_pulse(self, from_input: MachineModule,
                      pulse_strength: PulseStrength) -> List[Tuple[MachineModule, PulseStrength]]:
        """Process a pulse of given strength."""
        # Ignore high pulses
        if pulse_strength == PulseStrength.HIGH:
            return []
        if pulse_strength == PulseStrength.LOW:
            # Flip the state
            self.state = (not self.state[0],)

            # If state now on, send a high pulse to the output, otherwise send a low pulse
            pulse_strength = PulseStrength.HIGH if self.state[0] else PulseStrength.LOW
            return [(output, pulse_strength) for output in self.output]

        raise ValueError(f'Flip-flop module {self.name} received a pulse of strength {pulse_strength}')


class ConjunctionModule(MachineModule):
    """Class to represent a conjunction module."""
    def __init__(self, name: str) -> None:
        """Initialise the conjunction module."""
        logging.log(logging.DEBUG, f'Initialising conjunction module {name}')
        super().__init__(name, conjunct_symbol, tuple(), tuple(), tuple())

    def add_input(self, input_module: MachineModule) -> None:
        """Add an input module."""
        super().add_input(input_module)
        self.state += (False,)  # Add a state to represent our memory of that new input


    def process_pulse(self, from_input: MachineModule,
                      pulse_strength: PulseStrength) -> List[Tuple[MachineModule, PulseStrength]]:
        """Process a pulse of given strength."""
        # Make sure the pulse is either high or low
        if pulse_strength not in (PulseStrength.HIGH, PulseStrength.LOW):
            raise ValueError(f'Conjunction module {self.name} received a pulse of strength {pulse_strength}')

        # Update the state corresponding to that input
        input_idx = self.input.index(from_input)
        self.state = self.state[:input_idx] + (pulse_strength == PulseStrength.HIGH,) + self.state[input_idx + 1:]

        # If all the state is True, send a low pulse to the output, otherwise send a high pulse
        pulse_strength = PulseStrength.LOW if all(self.state) else PulseStrength.HIGH
        return [(output, pulse_strength) for output in self.output]


class ObserverModule(MachineModule):
    """Class to represent an observer module (it does nothing)."""
    def __init__(self, name: str) -> None:
        """Initialise the observer module."""
        logging.log(logging.DEBUG, f'Initialising observer module {name}')
        super().__init__(name, 'observer', tuple(), tuple(), tuple())

    def process_pulse(self, from_input: MachineModule,
                      pulse_strength: PulseStrength) -> List[Tuple[MachineModule, PulseStrength]]:
        """Process a pulse of given strength."""
        # Ignore the pulse
        return []


class MachineOnInterrupt(Exception):
    """Exception to be raised when the machine turns on."""
    pass


class Machine:
    """Class to represent the machine."""
    def __init__(self, schematic: List[str]) -> None:
        """Initialise the machine."""
        logging.log(logging.DEBUG, f'Initialising machine')
        self._module_map = {}  # Map from module name to module
        self._parse_schematic(schematic)

        self.num_low_pulses = 0
        self.num_high_pulses = 0

    def _parse_schematic(self, schematic: List[str]) -> None:
        """Set up a machine from a schematic."""
        # Split into module info and output info
        module_info = ['button']  # Button is always implicit
        output_info = ['broadcaster']
        for line in schematic:
            module, output = line.split(input_output_split_symbol)
            module_info.append(module)
            output_info.append(output)

        # Start by creating modules
        module_names = []
        for module_details in module_info:
            module_details = module_details.strip()

            # Check for special cases
            if module_details == 'button':
                module = ButtonModule()

            elif module_details == 'broadcaster':
                module = BroadcasterModule()

            # Otherwise, identify the module type and create it
            elif flip_flop_symbol in module_details:
                module = FlipFlopModule(module_details[1:])

            elif conjunct_symbol in module_details:
                module = ConjunctionModule(module_details[1:])

            else:
                raise ValueError(f'Unknown module type {module_details}')

            # Add the module to the map and list of names
            module_names.append(module.name)
            self._module_map[module.name] = module

        # Now add the input to output links
        for input_name, output_names in zip(module_names, output_info):
            # Split the output names
            output_names = output_names.strip().split(output_delimiter)
            output_names = [output_name.strip() for output_name in output_names]

            # Add the links
            for output_name in output_names:
                # Add observer module if necessary
                if output_name not in self._module_map.keys():
                    self._module_map[output_name] = ObserverModule(output_name)
                self._module_map[input_name].add_output(self._module_map[output_name])
                self._module_map[output_name].add_input(self._module_map[input_name])


    def press_the_button(self, flag_when_on: bool = False) -> None:
        """Press the button."""
        # Get the button module
        button = self._module_map['button']

        # Represent press by an initial pulse
        pulses = [(None, (button, PulseStrength.HIGH))]

        # Process the pulses
        while pulses:
            # Get the next pulse
            pulse = pulses.pop(0)

            # Process the pulse
            sender, pulse = pulse
            recipient, pulse_strength = pulse
            new_pulses = recipient.process_pulse(sender, pulse_strength)

            # Add to total
            for _, pulse in new_pulses:
                if pulse == PulseStrength.LOW:
                    self.num_low_pulses += 1
                elif pulse == PulseStrength.HIGH:
                    self.num_high_pulses += 1

            # Add new pulses to the queue
            for new_pulse in new_pulses:
                pulses += [(recipient, new_pulse)]


    def press_the_button_n_times(self, n: int) -> None:
        """Press the button n times."""
        for idx in range(n):
            self.press_the_button()


    def reset(self) -> None:
        """Reset the machine."""
        for module in self._module_map.values():
            module.state = tuple(False for _ in module.state)
        self.num_low_pulses = 0
        self.num_high_pulses = 0


    def num_potential_states(self) -> int:
        """Get the number of potential states the machine could have."""
        num_state_items = 0
        for module in self._module_map.values():
            num_state_items += len(module.state)
        return 2 ** num_state_items


# IO
def load_input(filename: str) -> List[str]:
    """Load the input from the given file."""
    logging.log(logging.DEBUG, f'Loading input from {filename}')
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() != ""]


def main():
    """Solve the puzzle!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    schematic = load_input(input_file)
    logging.log(logging.DEBUG, f'Loaded schematic with {len(schematic)} lines')

    # Set up the machine
    machine = Machine(schematic)

    # Part I, find the product of number of low and high pulses after 1000 button presses
    machine.press_the_button_n_times(1000)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 20 part 1, is: '
                              f'{machine.num_low_pulses * machine.num_high_pulses}')

    # Part II, find the number of button presses required to turn on the machine
    logging.log(logging.DEBUG, f'Resetting machine')
    machine.reset()

    # Can we brute force it?
    logging.log(logging.DEBUG, f'Number of potential states: {machine.num_potential_states()}')
    # Nope, too many states!

    """We need to find a way to understand the machine, so we can predict when it will turn on.
    
    On paper circuit analysis shows that the machine is made of four branches, and turns on only when all four
    branches are on simultaneously. The branches are each a counter circuit made of a chain of flip-flop modules
    and a single conjunction module (there is also a small post processing circuit to combine the branch outputs). 
    Within each counter the flip flops represent the binary digits, with there states storing together the
    current count value. 
    
    Each counter is calibrated to a different value, it will turn on after that many button presses, then resets to off
    and starts counting again. This calibration value is determined by the connection of the flip flops to the
    conjunction module. If a flip flop takes the conjunction module as input, then it is not included in the
    calibrated value, if it outputs to the conjunction module, then it is included in the calibrated value.
    We can thus read off the schematic the calibration values for each counter, e.g. how many times the button
    needs to be pressed before the counter will turn on (it will loop back to zero after that)."""
    counter_values = [4013, 3917, 3889, 3769]

    # The number of button presses required to turn on the machine is the lowest common multiple of the calibration
    # values for each counter.
    num_presses = 1
    for counter_value in counter_values:
        num_presses *= counter_value // gcd(num_presses, counter_value)
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 20 part 2, is: '
                              f'{num_presses}')


if __name__ == '__main__':
    main()

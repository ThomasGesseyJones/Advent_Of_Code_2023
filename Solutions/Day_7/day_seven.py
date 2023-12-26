"""Day Seven of Advent of Code 2023."""

# Import required packages
import logging
from typing import List
from dataclasses import dataclass
from functools import total_ordering

# Parameters
input_file = 'input.txt'


# Constants
card_order_wo_jokers = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
card_order_w_jokers = ['A', 'K', 'Q', 'T', '9', '8', '7', '6', '5', '4', '3', '2', 'J']


# Hand evaluation functions
def is_five_of_a_kind(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is five of a kind."""
    if not jokers_included:
        return all([card == cards[0] for card in cards])
    else:
        card_set = set(cards)
        if len(card_set) == 1:
            return True
        elif len(card_set) == 2 and 'J' in card_set:
            return True
        return False


def is_four_of_a_kind(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is four of a kind."""
    if not jokers_included:
        return any([cards.count(card) >= 4 for card in cards])
    else:
        return any([cards.count(card) + cards.count('J') >= 4 if card != 'J' else
                    cards.count('J') >= 4 for card in cards])


def is_full_house(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is a full house."""
    if not jokers_included:
        return any([cards.count(card) == 3 for card in cards]) and any([cards.count(card) == 2 for card in cards])
    else:
        if len(set(cards)) <= 3 and 'J' in cards:
            return True
        if len(set(cards)) <= 2 and 'J' not in cards:
            return is_full_house(cards, False)
        return False


def is_three_of_a_kind(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is three of a kind."""
    if not jokers_included:
        return any([cards.count(card) >= 3 for card in cards])
    else:
        return any([cards.count(card) + cards.count('J') >= 3 if card != 'J' else
                    cards.count('J') >= 3 for card in cards])


def is_two_pair(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is two pair."""
    if not jokers_included:
        return sum([cards.count(card) >= 2 for card in cards]) >= 4
    else:
        if 'J' not in cards:
            return is_two_pair(cards, False)
        if len(set(cards)) == 5:  # Can be shown by elimination, this is only possibility with jokers for no two pair
            return False
        return True


def is_one_pair(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is one pair."""
    if not jokers_included:
        return any([cards.count(card) >= 2 for card in cards])
    else:
        return any([cards.count(card) + cards.count('J') >= 2 if card != 'J' else
                    cards.count('J') >= 2 for card in cards])


def is_high_card(cards: List[str], jokers_included: bool) -> bool:
    """Check if a hand is a high card. It is always by definition."""
    return True


hand_order = [is_five_of_a_kind, is_four_of_a_kind, is_full_house, is_three_of_a_kind,
              is_two_pair, is_one_pair, is_high_card]


# IO
def load_input() -> List[str]:
    """Load the input file."""
    with open(input_file, 'r') as f:
        return [line.strip() for line in f.readlines()]


# Class to include properties of a hand and allow easy sorting
@total_ordering
@dataclass
class PokerHand:
    cards: List[str]
    bid: int
    jokers_include: bool = False

    @property
    def card_order(self) -> List[str]:
        """Get the card order for the current game."""
        if self.jokers_include:
            return card_order_w_jokers
        else:
            return card_order_wo_jokers


    def get_hand_type_rank(self) -> int:
        """Get the type rank of the hand.

        Lower is better.
        """
        for idx, hand_func in enumerate(hand_order):
            if hand_func(self.cards, self.jokers_include):
                return idx

    def _secondary_ordering_eq(self, other: 'PokerHand'):
        """Secondary ordering equality helper function."""
        return all(s == o for s, o in zip(self.cards, other.cards))

    def _secondary_ordering_gt(self, other: 'PokerHand'):
        """Secondary ordering greater than helper function."""
        for s, o in zip(self.cards, other.cards):
            if s != o:
                return self.card_order.index(s) < self.card_order.index(o)
        return False  # Equal hands are not greater than

    def __eq__(self, other: 'PokerHand') -> bool:
        """Equality operator."""
        if self.jokers_include != other.jokers_include:
            raise ValueError('Cannot compare hands with different joker inclusions.')
        if self.get_hand_type_rank() == other.get_hand_type_rank():
            return self._secondary_ordering_eq(other)
        return False

    def __gt__(self, other: 'PokerHand') -> bool:
        """Greater than operator."""
        if self.jokers_include != other.jokers_include:
            raise ValueError('Cannot compare hands with different joker inclusions.')
        if self.get_hand_type_rank() == other.get_hand_type_rank():
            return self._secondary_ordering_gt(other)
        return self.get_hand_type_rank() < other.get_hand_type_rank()


# Data Processing
def convert_input_to_hands(input_data: List[str]) -> List[PokerHand]:
    """Convert the input data to a list of poker hands."""
    logging.log(logging.DEBUG, f'Converting input data to poker hands.')
    hands = []
    for line in input_data:
        cards, bid = line.split()
        hands.append(PokerHand([c for c in cards], int(bid)))
    return hands


def main():
    """Solve the problem!"""
    logging.basicConfig(level=logging.INFO)

    # Load the input
    input_data = load_input()
    card_hands = convert_input_to_hands(input_data)
    logging.log(logging.DEBUG, f'Loaded {len(card_hands)} hands.')

    # Sort the hands
    logging.log(logging.DEBUG, f'Sorting hands.')
    card_hands.sort()
    logging.log(logging.DEBUG, f'{card_hands=}')

    # Calculate winnings
    winnings = [hand.bid*(idx + 1) for idx, hand in enumerate(card_hands)]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 7 part 1, is: {sum(winnings)}')

    # Turn jokers on and recalculate
    logging.log(logging.DEBUG, f'Turning jokers on.')
    for hand in card_hands:
        hand.jokers_include = True
    card_hands.sort()
    logging.log(logging.DEBUG, f'{card_hands=}')
    winnings = [hand.bid*(idx + 1) for idx, hand in enumerate(card_hands)]
    logging.log(logging.INFO, f'The answer to Advent of Code 2023, day 7 part 2, is: {sum(winnings)}')


if __name__ == '__main__':
    main()

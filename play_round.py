import numpy as np
from dataclasses import dataclass
from branch_calc import (
    filter_branch_by_hand,
    setdiff2d_idx,
    find_winner,
    calc_all_possible_hands,
)
@dataclass
class PlayRound:
    hands: list
    winner: int

    def play_round(self, bit: int):
        all_possible_tricks = calc_all_possible_hands(hands=self.hands)
        len(all_possible_tricks)
        contains_target = np.any(
            np.all(all_possible_tricks == self.hands[self.winner][bit], axis=-1), axis=1
        )
        stump = all_possible_tricks[contains_target]
        stump = filter_branch_by_hand(stump, self.hands[(self.winner+1)%4], (self.winner+1)%4, self.hands[self.winner][bit])
        stump = filter_branch_by_hand(stump, self.hands[(self.winner+2)%4], (self.winner+2)%4, self.hands[self.winner][bit])
        stump = filter_branch_by_hand(stump, self.hands[(self.winner+3)%4], (self.winner+3)%4, self.hands[self.winner][bit])
        self.round_winner = find_winner(
            lead=stump[bit][self.winner], trick=stump[bit]
        )  # will need to fix this in order to iterate through all hands
        self.round_winning_team = (
            find_winner(lead=stump[bit][self.winner], trick=stump[bit]) % 2
        )  # will need to fix this in order to iterate through all hands
        self.next_round_hands = [
            setdiff2d_idx(self.hands[0], stump[bit]),
            setdiff2d_idx(self.hands[1], stump[bit]),
            setdiff2d_idx(self.hands[2], stump[bit]),
            setdiff2d_idx(self.hands[3], stump[bit]),
        ]

        return (
            stump[bit],
            self.round_winner,
            self.round_winning_team,
            np.array(self.next_round_hands),
        )
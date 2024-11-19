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
    lead: int
    card_play: int

    def play_round(self, bit: int):
        all_possible_tricks = calc_all_possible_hands(hands=self.hands)
        len(all_possible_tricks)
        contains_target = np.any(
            np.all(all_possible_tricks == self.hands[self.lead][self.card_play], axis=-1), axis=1
        )
        stump = all_possible_tricks[contains_target]
        stump = filter_branch_by_hand(stump, self.hands[(self.lead+1)%4], (self.lead+1)%4, self.hands[self.lead][self.card_play])
        stump = filter_branch_by_hand(stump, self.hands[(self.lead+2)%4], (self.lead+2)%4, self.hands[self.lead][self.card_play])
        stump = filter_branch_by_hand(stump, self.hands[(self.lead+3)%4], (self.lead+3)%4, self.hands[self.lead][self.card_play])
        self.round_winner = find_winner(
            lead=stump[bit][self.lead], trick=stump[bit]
        )  # will need to fix this in order to iterate through all hands
           # this appears to actually work so far
        self.round_winning_team = (
            find_winner(lead=stump[bit][self.lead], trick=stump[bit]) % 2
        )  # will need to fix this in order to iterate through all hands
           # this appears to actually work so far
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
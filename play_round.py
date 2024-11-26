import numpy as np
from dataclasses import dataclass
from branch_calc import (
    filter_branch_by_hand,
    calc_all_possible_hands,
)
@dataclass
class PlayRound:
    hands: list
    lead: int
    card_play: int

    def play_round(self):
        all_possible_tricks = calc_all_possible_hands(hands=self.hands)
        len(all_possible_tricks)
        contains_target = np.any(
            np.all(all_possible_tricks == self.hands[self.lead][self.card_play], axis=-1), axis=1
        )
        stump = all_possible_tricks[contains_target]
        stump = filter_branch_by_hand(stump, self.hands[(self.lead+1)%4], (self.lead+1)%4, self.hands[self.lead][self.card_play])
        stump = filter_branch_by_hand(stump, self.hands[(self.lead+2)%4], (self.lead+2)%4, self.hands[self.lead][self.card_play])
        stump = filter_branch_by_hand(stump, self.hands[(self.lead+3)%4], (self.lead+3)%4, self.hands[self.lead][self.card_play])
        return stump
    
def find_winners_vectorized(leads, tricks):
# Calculate norms for all cards in all tricks at once
    norms = np.linalg.norm(tricks, axis=2)  # Shape: (N, 4)
    
    # Check if any trick has a card with norm > 80
    if np.any(norms > 80):
        # For high value tricks, winner is card with maximum norm
        winners = np.argmax(norms, axis=1)
        return winners

    # identifies the suit of the vector by its angle from the positive x asix
    suit_ids = np.arctan2(tricks[:, :, 1], tricks[:, :, 0])
    
    # Get the suit IDs of the leads
    lead_suits = suit_ids[np.arange(len(leads)), leads]
    
    # Create mask for matching suits (using small epsilon for float comparison)
    matching_suits = np.abs(suit_ids - lead_suits[:, np.newaxis]) < 1e-10  # Shape: (N, 4)
    
    # Set norms of non-matching suits to -inf
    masked_norms = np.where(matching_suits, norms, -np.inf)
    
    # Find winners for regular tricks
    winners = np.argmax(masked_norms, axis=1)
    
    return winners
    

    # def next_round_hands(results: np.ndarray):
    #     next_round_hands = [
    #         setdiff2d_idx(self.hands[0], stump[bit]),
    #         setdiff2d_idx(self.hands[1], stump[bit]),
    #         setdiff2d_idx(self.hands[2], stump[bit]),
    #         setdiff2d_idx(self.hands[3], stump[bit]),
    #     ]
    



           # will need to fix this in order to iterate through all hands
           # this appears to actually work so far
# def next_round_hands(results: np.ndarray)
#         next_round_hands = [
#             setdiff2d_idx(self.hands[0], stump[bit]),
#             setdiff2d_idx(self.hands[1], stump[bit]),
#             setdiff2d_idx(self.hands[2], stump[bit]),
#             setdiff2d_idx(self.hands[3], stump[bit]),
#         ]

#         return (
#             stump[bit],
#             self.round_winner,
#             self.round_winning_team,
#             np.array(self.next_round_hands),
#         )
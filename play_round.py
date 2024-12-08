import numpy as np
from numba import njit
from branch_calc import (
    filter_branch_by_hand,
    calc_all_possible_hands,
)

def play_round(hands, lead, card_play):
    all_possible_tricks = calc_all_possible_hands(hands=hands)
    contains_target = np.any(
        np.all(all_possible_tricks == hands[lead][card_play], axis=-1), axis=1
    )
    stump = all_possible_tricks[contains_target]
    stump = filter_branch_by_hand(stump, hands[(lead+1)%4], (lead+1)%4, hands[lead][card_play])
    stump = filter_branch_by_hand(stump, hands[(lead+2)%4], (lead+2)%4, hands[lead][card_play])
    stump = filter_branch_by_hand(stump, hands[(lead+3)%4], (lead+3)%4, hands[lead][card_play]) 
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

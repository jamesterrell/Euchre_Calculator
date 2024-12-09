from numba import njit
import numpy as np
from n_branches import (
    n_tricks,
    n_ap_filter,
    nfb_by_hand,
    n_find_winner,
    array_set_difference
)

@njit
def n_play_round(hands, lead, card_play):
    all_possible_tricks = n_tricks(hands)
    branch = n_ap_filter(tricks=all_possible_tricks, target=hands[lead][card_play], lead=lead)
    branch = nfb_by_hand(branch=branch, hand=(lead+1)%4, target=hands[lead][card_play])
    branch = nfb_by_hand(branch=branch, hand=(lead+2)%4, target=hands[lead][card_play])
    branch = nfb_by_hand(branch=branch, hand=(lead+3)%4, target=hands[lead][card_play])
    return branch

@njit
def round1(hands_dealt):
    # Pre-allocate maximum possible size
    max_branches = 625  # Worst-case scenario
    r2_hands = np.zeros((max_branches, hands_dealt.shape[0], hands_dealt.shape[1]-1, hands_dealt.shape[2]), dtype=np.int64)
    r2_leads = np.zeros(max_branches, dtype=np.int64)
    
    # Track actual number of branches
    branch_count = 0
    
    for card in range(5):
        branch = n_play_round(hands=hands_dealt, lead=0, card_play=card)
        
        for i in branch:
            hand_set = array_set_difference(hands_dealt, i)
            
            # Directly assign to pre-allocated array
            r2_hands[branch_count, :, :, :] = hand_set
            rd1_winner = n_find_winner(trick=i, lead=0)
            r2_leads[branch_count] = rd1_winner
            
            branch_count += 1
    print(branch_count)
    # Trim to actual size
    return r2_leads[:branch_count], r2_hands[:branch_count, :, :, :]
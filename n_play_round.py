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

@njit
def next_round(current_hands, leads, game_round, game_score):
    # Pre-allocate maximum possible size
    max_branches = len(leads)*4*4*4*4  # Worst-case scenario
    next_round_hands = np.zeros(shape=(max_branches, current_hands.shape[1], current_hands.shape[2]-1, current_hands.shape[3]), dtype=np.int64)
    next_round_leads= np.zeros(max_branches, dtype=np.int64)
    total_score = np.zeros(shape=(max_branches, game_round, 1), dtype=np.int64)
    possible_play = 5-game_round+1
    # Track actual number of branches
    branch_count = 0

    for hand, lead, score in zip(current_hands, leads, game_score):
        for card in range(possible_play):
            branch = n_play_round(hands=hand, lead=lead, card_play=card)
            for i in branch:
                hand_set = array_set_difference(hand, i)
                next_round_hands[branch_count, :, :, :] = hand_set
                rd_winner = n_find_winner(trick=i, lead=lead)
                next_round_leads[branch_count] = rd_winner
                # try:
                # for j in range(len(total_score[branch_count])-1):
                #     total_score[branch_count, j] = score
                # except:
                for j in range(game_round-1):
                    total_score[branch_count, j] = score[j]

                
                total_score[branch_count, game_round-1] = rd_winner
                
                branch_count += 1
    print(branch_count)
    # Trim to actual size
    return next_round_leads[:branch_count], next_round_hands[:branch_count, :, :, :], total_score[:branch_count]
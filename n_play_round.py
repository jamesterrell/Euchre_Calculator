from numba import njit
import numpy as np
from n_branches import (
    n_tricks,
    n_ap_filter,
    nfb_by_hand,
    n_find_winner,
    array_set_difference,
    common_sense,
    smart_loss,
    trump_or_dump
)

@njit
def n_play_round(hands, lead, card_play):
    """
    Simulates a single round of play in a card game where each player plays one card.
    The function calculates all possible tricks, applies various filters (such as filtering 
    by hand and common sense), and applies logic for deciding to play a trump or the worst card. 
    The final branch representing all possible outcomes of the round is returned.

    Arguments:
        hands (numpy.ndarray): A 3D array representing the cards dealt to each player.
        lead (int): The index of the player who leads the trick.
        card_play (int): The index of the card played by the lead player.

    Returns:
        numpy.ndarray: A 3D array representing all possible hands after the round.
    """
    all_possible_tricks = n_tricks(hands)
    branch = n_ap_filter(tricks=all_possible_tricks, target=hands[lead][card_play], lead=lead)

    branch = nfb_by_hand(branch=branch, hand=(lead+1)%4, target=hands[lead][card_play])
    branch = common_sense(branch=branch, target=hands[lead][card_play], player=(lead+1)%4)
    branch = smart_loss(branch=branch, target=hands[lead][card_play], player=(lead+1)%4)
    branch = trump_or_dump(branch=branch, target=hands[lead][card_play], player=(lead+1)%4) 

    branch = nfb_by_hand(branch=branch, hand=(lead+2)%4, target=hands[lead][card_play])
    branch = common_sense(branch=branch, target=hands[lead][card_play], player=(lead+2)%4)
    branch = smart_loss(branch=branch, target=hands[lead][card_play], player=(lead+2)%4)
    branch = trump_or_dump(branch=branch, target=hands[lead][card_play], player=(lead+2)%4)

    branch = nfb_by_hand(branch=branch, hand=(lead+3)%4, target=hands[lead][card_play])
    branch = common_sense(branch=branch, target=hands[lead][card_play], player=(lead+3)%4)
    branch = smart_loss(branch=branch, target=hands[lead][card_play], player=(lead+3)%4)
    branch = trump_or_dump(branch=branch, target=hands[lead][card_play], player=(lead+3)%4)

    return branch

@njit
def round1(hands_dealt):
    """
    Simulates the first round of the game, generating possible outcomes after each player 
    plays one card. The function tracks the winner of each trick, updates the hands based on 
    the played cards, and returns the winner's leads and the updated hands after the round.

    Arguments:
        hands_dealt (numpy.ndarray): A 3D array representing the hands dealt to each player.

    Returns:
        tuple: A tuple containing:
            - r2_leads (numpy.ndarray): The array of winners for each round.
            - r2_hands (numpy.ndarray): The array of hands after each round.
    """
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
    # Trim to actual size
    return r2_leads[:branch_count], r2_hands[:branch_count, :, :, :]

@njit
def next_round(current_hands, leads, game_round, game_score):
    """
    Simulates the next round of the game based on the current hands and scores. It generates 
    all possible outcomes after each player plays a card and updates the scores and hands accordingly. 
    The function tracks the outcome of the trick and the updated hands, and returns the next round's leads, 
    hands, and cumulative scores.

    Arguments:
        current_hands (numpy.ndarray): A 3D array representing the current hands of all players.
        leads (numpy.ndarray): The current lead players for each trick.
        game_round (int): The current round number in the game.
        game_score (numpy.ndarray): The cumulative scores of each player.

    Returns:
        tuple: A tuple containing:
            - next_round_leads (numpy.ndarray): The array of winners for each trick.
            - next_round_hands (numpy.ndarray): The array of hands after each trick.
            - total_score (numpy.ndarray): The cumulative scores after each trick.
"""
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
    # Trim to actual size
    return next_round_leads[:branch_count], next_round_hands[:branch_count, :, :, :], total_score[:branch_count]
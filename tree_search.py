from n_play_round import round1, next_round
import numpy as np
from numba import njit
from typing import Callable

@njit
def n_trick_sim(
    game_hand: np.ndarray, 
    r1_chosen_card: np.ndarray, 
    lead: int,
    num_tricks: int = 5,
    previous_winners: np.ndarray = np.array([], dtype=np.int64)
):
    """
    Simulates a game with a specified number of tricks and evaluates the outcome.

    Arguments:
        game_hand (numpy.ndarray): A 3D array representing the cards dealt to each player.
        r1_chosen_card (np.ndarray): The index of the card chosen for the first round.
        lead (int): The player who leads the first trick.
        num_tricks (int): Number of tricks to simulate (1-5). Default is 5.
        previous_winners (np.ndarray): Optional array of previous trick winners to include in scoring.

    Returns:
        float: The mean performance evaluation, indicating win probability.
    """
    
    # First round
    r2_leads, r2_hands = round1(
        hands_dealt=game_hand, chosen_card=r1_chosen_card, leader=lead
    )
    
    # Handle different numbers of tricks explicitly
    if num_tricks == 1:
        results = r2_leads.reshape(-1, 1)
    
    elif num_tricks == 2:
        r3_leads, r3_hands, r3_score = next_round(
            current_hands=r2_hands,
            leads=r2_leads,
            game_round=5,
            game_score=r2_leads.reshape(-1, 1)
        )
        results = r3_score.reshape(r3_score.shape[0], 5)
    
    elif num_tricks == 3:
        r3_leads, r3_hands, r3_score = next_round(
            current_hands=r2_hands,
            leads=r2_leads,
            game_round=4,
            game_score=r2_leads.reshape(-1, 1)
        )
        r4_leads, r4_hands, r4_score = next_round(
            current_hands=r3_hands,
            leads=r3_leads,
            game_round=5,
            game_score=r3_score
        )
        results = r4_score.reshape(r4_score.shape[0], 5)
    
    elif num_tricks == 4:
        r3_leads, r3_hands, r3_score = next_round(
            current_hands=r2_hands,
            leads=r2_leads,
            game_round=3,
            game_score=r2_leads.reshape(-1, 1)
        )
        r4_leads, r4_hands, r4_score = next_round(
            current_hands=r3_hands,
            leads=r3_leads,
            game_round=4,
            game_score=r3_score
        )
        r5_leads, r5_hands, r5_score = next_round(
            current_hands=r4_hands,
            leads=r4_leads,
            game_round=5,
            game_score=r4_score
        )
        results = r5_score.reshape(r5_score.shape[0], 5)

    
    else:  # num_tricks == 5
        r3_leads, r3_hands, r3_score = next_round(
            current_hands=r2_hands,
            leads=r2_leads,
            game_round=2,
            game_score=r2_leads.reshape(-1, 1)
        )
        r4_leads, r4_hands, r4_score = next_round(
            current_hands=r3_hands,
            leads=r3_leads,
            game_round=3,
            game_score=r3_score
        )
        r5_leads, r5_hands, r5_score = next_round(
            current_hands=r4_hands,
            leads=r4_leads,
            game_round=4,
            game_score=r4_score
        )
        r6_leads, r6_hands, r6_score = next_round(
            current_hands=r5_hands,
            leads=r5_leads,
            game_round=5,
            game_score=r5_score
        )
        results = r6_score.reshape(r6_score.shape[0], 5)
    
    # Calculate meta results
    meta_results = np.zeros(results.shape[0], dtype=np.int64)
    
    for i in range(len(results)):
        # Count wins for odd-numbered players (team 1)
        total_odd_wins = np.sum(results[i] % 2)
        
        # Add previous winners to the count
        if len(previous_winners) > 0:
            total_odd_wins += np.sum(previous_winners % 2)
        
        # Team wins if they get 3+ tricks out of 5 total
        if total_odd_wins >= 3:
            score = 0
        else: 
            score = 1

        meta_results[i] = score

    return np.mean(meta_results)

def find_best_opener(
    hands: np.ndarray, lead: int, tricks: int, previous_winners: np.array, sim_func: Callable, 
):
    winning_chances = np.zeros(tricks)
    for i in range(tricks):
        winning_chances[i] = sim_func(
            game_hand=hands, r1_chosen_card=i, lead=lead, num_tricks=tricks, previous_winners=previous_winners
        )

    print(winning_chances)

    if lead % 2 == 0:
        best_opener = np.argmax(winning_chances)

    else:
        best_opener = np.argmin(winning_chances)

    return best_opener


@njit
def trick_played(arr1, arr2):
    # Flatten both arrays to shape (N, 2)
    flat1 = arr1.reshape(-1, arr1.shape[-1])
    flat2 = arr2.reshape(-1, arr2.shape[-1])

    # Preallocate output array (worst case: all from arr1 are missing)
    missing_rows = np.empty((flat1.shape[0], flat1.shape[1]), dtype=arr1.dtype)
    count = 0

    for i in range(flat1.shape[0]):
        found = False
        for j in range(flat2.shape[0]):
            if flat1[i, 0] == flat2[j, 0] and flat1[i, 1] == flat2[j, 1]:
                found = True
                break
        if not found:
            missing_rows[count] = flat1[i]
            count += 1

    return missing_rows[:count]  


@njit
def r2_best_response(hands, best_opening_card):
    r2_leads, r2_hands = round1(hands_dealt=hands, chosen_card=best_opening_card)
    print(r2_hands)
    resp_win_chances = np.zeros(len(r2_hands))
    counter = 0
    for hand, lead in zip(r2_hands, r2_leads):
        r3_leads, r3_hands, r2_score = next_round(
            current_hands=hand.reshape(1, 4, 4, 2),
            leads=np.array([lead]),
            game_round=2,
            game_score=np.array([lead]).reshape(-1, 1),
        )
        r4_leads, r4_hands, r3_score = next_round(
            current_hands=r3_hands, leads=r3_leads, game_round=3, game_score=r2_score
        )
        r5_leads, r5_hands, r4_score = next_round(
            current_hands=r4_hands, leads=r4_leads, game_round=4, game_score=r3_score
        )
        r6_leads, r6_hands, r5_score = next_round(
            current_hands=r5_hands, leads=r5_leads, game_round=5, game_score=r4_score
        )
        results = r5_score.reshape(r5_score.shape[0], 5)

        meta_results = np.zeros(results.shape[0], dtype=np.int64)

        eval_position = 1

        if eval_position % 2 == 0:
            for i in range(len(results)):
                meta_results[i] = np.sum(results[i] % 2) < 3
        else:
            for i in range(len(results)):
                meta_results[i] = np.sum(results[i] % 2) >= 3

        resp_win_chances[counter] = np.mean(meta_results)
        counter += 1

    print(resp_win_chances)
    best_response = np.argmax(resp_win_chances)
    return r2_hands[best_response], r2_leads[best_response]

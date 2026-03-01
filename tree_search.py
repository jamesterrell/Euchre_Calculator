from n_play_round import round1, next_round
import numpy as np
from numba import njit
from typing import Callable

@njit
def declare_winner(winners: np.ndarray):
    """
    Determines which team won based on trick winners.
    
    Arguments:
        winners (np.ndarray): 1D array of 5 player indices indicating who won each trick.
                              Example: [1, 2, 1, 0, 3]
        
    Returns:
        int: 0 if odd team (positions 1, 3) won, 1 if even team (positions 0, 2) won
    """
    # Count wins for odd-numbered players (team 1)
    total_odd_wins = np.sum(winners % 2)
    
    # Odd team wins if they get 3+ tricks out of 5 total
    if total_odd_wins >= 3:
        score = 0  # Odd team (1, 3) won
    else: 
        score = 1  # Even team (0, 2) won
    
    return score

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
    hands: np.ndarray, lead: int, tricks: int, previous_winners: np.array, sim_func: Callable, verbose: bool
):
    winning_chances = np.zeros(tricks)
    for i in range(tricks):
        winning_chances[i] = sim_func(
            game_hand=hands, r1_chosen_card=i, lead=lead, num_tricks=tricks, previous_winners=previous_winners
        )

    if verbose:
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


def find_best_response(
    hands: np.ndarray,
    lead: int,
    tricks: int,
    previous_winners: np.ndarray,
    best_opener: int,
    verbose: bool
):
    """
    Determines the optimal response card for the opposing team after the lead player 
    has played their opening card.

    This function simulates all possible responses to the opening card and evaluates 
    which response maximizes the responding team's chances of winning. It considers 
    the full game tree from the current state through all remaining tricks.

    Arguments:
        hands (np.ndarray): A 3D array of shape (4, n, 2) representing the current hands 
            for all four players, where n is the number of cards remaining.
        lead (int): The index (0-3) of the player who led the trick.
        tricks (int): The number of tricks remaining to be played (1-5).
        previous_winners (np.ndarray): Array of previous trick winners for scoring.
        best_opener (int): The index of the card the lead player chose to play.

    Returns:
        tuple: A tuple containing:
            - optimal_response (np.ndarray): The hand configuration after the optimal response
            - winner (int): The index of the player who won this trick
    """
    
    responder = (lead + 1) % 4  # The next player after lead
    
    # Simulate the first trick with the chosen opening card
    r2_leads, r2_hands = round1(
        hands_dealt=hands, chosen_card=best_opener, leader=lead
    )
    
    r1_response_res = np.zeros(r2_leads.shape, dtype=np.float64)
    
    if tricks == 5:
        # 5 tricks: Simulate all 4 remaining rounds
        for w in range(r2_leads.shape[0]):
            r3_leads, r3_hands, r2_score = next_round(
                current_hands=np.array([r2_hands[w]]),
                leads=np.array([r2_leads[w]]),
                game_round=2,
                game_score=np.array([r2_leads[w]]).reshape(-1, 1),
            )
            r4_leads, r4_hands, r3_score = next_round(
                current_hands=r3_hands,
                leads=r3_leads,
                game_round=3,
                game_score=r2_score,
            )
            r5_leads, r5_hands, r4_score = next_round(
                current_hands=r4_hands,
                leads=r4_leads,
                game_round=4,
                game_score=r3_score,
            )
            r6_leads, r6_hands, r5_score = next_round(
                current_hands=r5_hands,
                leads=r5_leads,
                game_round=5,
                game_score=r4_score,
            )
            
            results = r5_score.reshape(r5_score.shape[0], 5)
            meta_results = np.zeros(results.shape[0], dtype=np.int64)
            
            for i in range(len(results)):
                total_odd_wins = np.sum(results[i] % 2) + np.sum(previous_winners % 2)
                meta_results[i] = total_odd_wins >= 3
            
            r1_response_res[w] = np.mean(meta_results)
    
    elif tricks == 4:
        # 4 tricks: Simulate 3 remaining rounds
        for w in range(r2_leads.shape[0]):
            r3_leads, r3_hands, r2_score = next_round(
                current_hands=np.array([r2_hands[w]]),
                leads=np.array([r2_leads[w]]),
                game_round=3,
                game_score=np.array([r2_leads[w]]).reshape(-1, 1),
            )
            r4_leads, r4_hands, r3_score = next_round(
                current_hands=r3_hands,
                leads=r3_leads,
                game_round=4,
                game_score=r2_score,
            )
            r5_leads, r5_hands, r4_score = next_round(
                current_hands=r4_hands,
                leads=r4_leads,
                game_round=5,
                game_score=r3_score,
            )
            
            results = r4_score.reshape(r4_score.shape[0], 5)
            meta_results = np.zeros(results.shape[0], dtype=np.int64)
            
            for i in range(len(results)):
                total_odd_wins = np.sum(results[i] % 2) + np.sum(previous_winners % 2)
                meta_results[i] = total_odd_wins >= 3
            
            r1_response_res[w] = np.mean(meta_results)
    
    elif tricks == 3:
        # 3 tricks: Simulate 2 remaining rounds
        for w in range(r2_leads.shape[0]):
            r3_leads, r3_hands, r2_score = next_round(
                current_hands=np.array([r2_hands[w]]),
                leads=np.array([r2_leads[w]]),
                game_round=4,
                game_score=np.array([r2_leads[w]]).reshape(-1, 1),
            )
            r4_leads, r4_hands, r3_score = next_round(
                current_hands=r3_hands,
                leads=r3_leads,
                game_round=5,
                game_score=r2_score,
            )
            
            results = r3_score.reshape(r3_score.shape[0], 5)
            meta_results = np.zeros(results.shape[0], dtype=np.int64)
            
            for i in range(len(results)):
                total_odd_wins = np.sum(results[i] % 2) + np.sum(previous_winners % 2)
                meta_results[i] = total_odd_wins >= 3
            
            r1_response_res[w] = np.mean(meta_results)
    
    elif tricks == 2:
        # 2 tricks: Simulate 1 remaining round
        for w in range(r2_leads.shape[0]):
            r3_leads, r3_hands, r2_score = next_round(
                current_hands=np.array([r2_hands[w]]),
                leads=np.array([r2_leads[w]]),
                game_round=5,
                game_score=np.array([r2_leads[w]]).reshape(-1, 1),
            )
            
            results = r2_score.reshape(r2_score.shape[0], 5)
            meta_results = np.zeros(results.shape[0], dtype=np.int64)
            
            for i in range(len(results)):
                total_odd_wins = np.sum(results[i] % 2) + np.sum(previous_winners % 2)
                meta_results[i] = total_odd_wins >= 3
            
            r1_response_res[w] = np.mean(meta_results)
    
    elif tricks == 1:
        # 1 trick: No more rounds to simulate, just evaluate this trick
        for w in range(r2_leads.shape[0]):
            results = np.array([r2_leads[w]]).reshape(1, 1)
            meta_results = np.zeros(results.shape[0], dtype=np.int64)
            
            for i in range(len(results)):
                total_odd_wins = np.sum(results[i] % 2) + np.sum(previous_winners % 2)
                meta_results[i] = total_odd_wins >= 3
            
            r1_response_res[w] = np.mean(meta_results)
    
    # Select best response based on which team the responder is on

    # I'm struggling to figure out what happend, but switching the argmin/max maybe fixed the the fucked up response logic. 
    # need to come back here and try to understand it. 
    if responder % 2 == 0:
        best_response = np.argmin(r1_response_res)  # Even team wants to min odd wins
    else:
        best_response = np.argmax(r1_response_res)  # Odd team wants to max odd wins

    

    optimal_response = r2_hands[best_response]
    winner = r2_leads[best_response]
    if verbose:
        print(f'Lead {lead}')
        print(f'Hands: \n{hands}')
        print(f'Trick Played: \n{trick_played(hands, r2_hands[best_response])}')
    
    return optimal_response, winner


def definitive_winner(given_hands: np.ndarray, verbose: bool):
    """
    Determines the definitive winner of a complete 5-trick Euchre game through optimal play.
    
    This function simulates a full game of Euchre where both teams play optimally at each 
    decision point. It uses a minimax-style approach, alternating between finding the best 
    opening card for the leading team and the best response from the opposing team for all 
    five tricks.
    
    The game proceeds as follows:
    1. Player 1 (odd team) finds their optimal opening card
    2. The opposing team's best response is calculated
    3. The winner of trick 1 leads trick 2, and the process repeats
    4. This continues for all 5 tricks
    5. The team that wins 3+ tricks is declared the winner
    
    Arguments:
        given_hands (np.ndarray): A 3D array of shape (4, 5, 2) representing the initial 
            hands dealt to all four players. Each card is represented as a 2D vector [x, y].
            Players are indexed 0-3, where:
                - Team 0 (even): players 0 and 2
                - Team 1 (odd): players 1 and 3
        verbose (bool): If True, prints detailed information about each trick including 
            cards played, win probabilities, and intermediate game states. If False, runs 
            silently.
    
    Returns:
        int: The winning team index:
            - 0 if the odd team (players 1, 3) won the game (3+ tricks)
            - 1 if the even team (players 0, 2) won the game (3+ tricks)
    """

    best_opener = find_best_opener(
        hands=given_hands, lead=1, tricks=5, previous_winners=np.array([]), sim_func=n_trick_sim, verbose=verbose
    )
    r2_optimal, winner_round1 = find_best_response(
        hands=given_hands,
        lead=1,
        tricks=5,
        previous_winners=np.array([]),
        best_opener=best_opener,
        verbose=verbose
    )
    r2_best_opener = find_best_opener(
        hands=r2_optimal,
        lead=winner_round1,
        tricks=4,
        previous_winners=np.array([winner_round1]),
        sim_func=n_trick_sim,
        verbose=verbose
    )
    r3_optimal, winner_round2 = find_best_response(
        hands=r2_optimal,
        lead=winner_round1,
        tricks=4,
        previous_winners=np.array([winner_round1]),
        best_opener=r2_best_opener,
        verbose=verbose

    )
    r3_best_opener = find_best_opener(
        hands=r3_optimal,
        lead=winner_round2,
        tricks=3,
        previous_winners=np.array([winner_round1, winner_round2]),
        sim_func=n_trick_sim,
        verbose=verbose
    )
    r4_optimal, winner_round3 = find_best_response(
        hands=r3_optimal,
        lead=winner_round2,
        tricks=3,
        previous_winners=np.array([winner_round1, winner_round2]),
        best_opener=r3_best_opener,
        verbose=verbose

    )
    r4_best_opener = find_best_opener(
        hands=r4_optimal,
        lead=winner_round3,
        tricks=2,
        previous_winners=np.array([winner_round1, winner_round2, winner_round3]),
        sim_func=n_trick_sim,
        verbose=verbose
    )
    r5_optimal, winner_round4 = find_best_response(
        hands=r4_optimal,
        lead=winner_round3,
        tricks=2,
        previous_winners=np.array([winner_round1, winner_round2, winner_round3]),
        best_opener=r4_best_opener,
        verbose=verbose

    )
    r5_best_opener = find_best_opener(
        hands=r5_optimal,
        lead=winner_round4,
        tricks=1,
        previous_winners=np.array(
            [winner_round1, winner_round2, winner_round3, winner_round4]
        ),
        sim_func=n_trick_sim,
        verbose=verbose
    )

    r6_optimal, winner_round5 = find_best_response(
        hands=r5_optimal,
        lead=winner_round4,
        tricks=1,
        previous_winners=np.array([winner_round1, winner_round2, winner_round3, winner_round4]),
        best_opener=r5_best_opener,
        verbose=verbose

    )

    total_winners = np.array([winner_round1, winner_round2, winner_round3, winner_round4, winner_round5], dtype=np.int64)

    result = declare_winner(winners=total_winners)

    return result

    

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


def player_best_prune(
    player: int,
    first_response: bool,
    best_opener: int = 0,
    starting_lead: int = 0,
    tricks: int = 5,
    starting_hands: np.ndarray = np.array([], dtype=np.int64),
    given_hands: np.ndarray = np.array([], dtype=np.int64),
    given_leads: np.ndarray = np.array([], dtype=np.int64),
    previous_winners: np.ndarray = np.array([], dtype=np.int64)
):
    """
    Prunes the game tree by filtering to only the branches where a specific player
    plays optimally for their team.
    
    Arguments:
        player (int): Which player position (0-3) to optimize for
        first_response (bool): If True, generate initial branches from round1. 
                               If False, use provided given_hands/given_leads
        best_opener (int): Index of opening card (used only if first_response=True)
        starting_lead (int): Player who led the current trick
        tricks (int): Number of tricks remaining (1-5)
        starting_hands (np.ndarray): Initial hand configurations (used if first_response=True)
        given_hands (np.ndarray): Current hand configurations (used if first_response=False)
        given_leads (np.ndarray): Current lead positions (used if first_response=False)
        previous_winners (np.ndarray): Array of previous trick winners for scoring
    
    Returns:
        tuple: (final_prune, leads_prune) - filtered hands and leads after pruning
    """
    
    if first_response:
        given_leads, given_hands = round1(
            hands_dealt=starting_hands, chosen_card=best_opener, leader=starting_lead
        )

    # responder = (starting_lead + player) % 4

    # Find all unique cards this player could play
    possible_plays_responder = np.unique(given_hands[:, player], axis=0)

    prune_masks = np.zeros(
        shape=(possible_plays_responder.shape[0], given_hands.shape[0]), dtype=np.bool_
    )
    best_pruned_branch_res = np.zeros(
        possible_plays_responder.shape[0], dtype=np.float64
    )

    pruned_idx = 0

    # Evaluate each possible card this player could play
    for target in possible_plays_responder:
        # Filter to branches where this player plays this specific card
        mask = np.all(given_hands[:, player] == target, axis=(1, 2))
        filtered_hands = given_hands[mask]
        filtered_leads = given_leads[mask]

        # Simulate remaining tricks based on how many are left
        if tricks == 5:
            # 5 tricks: Simulate rounds 2, 3, 4, 5
            r3_leads, r3_hands, r3_score = next_round(
                current_hands=filtered_hands,
                leads=filtered_leads,
                game_round=2,
                game_score=filtered_leads.reshape(-1, 1),
            )
            r4_leads, r4_hands, r4_score = next_round(
                current_hands=r3_hands, leads=r3_leads, game_round=3, game_score=r3_score
            )
            r5_leads, r5_hands, r5_score = next_round(
                current_hands=r4_hands, leads=r4_leads, game_round=4, game_score=r4_score
            )
            r6_leads, r6_hands, r6_score = next_round(
                current_hands=r5_hands, leads=r5_leads, game_round=5, game_score=r5_score
            )
            results = r6_score.reshape(r6_score.shape[0], 5)
        
        elif tricks == 4:
            # 4 tricks: Simulate rounds 3, 4, 5
            r3_leads, r3_hands, r3_score = next_round(
                current_hands=filtered_hands,
                leads=filtered_leads,
                game_round=3,
                game_score=filtered_leads.reshape(-1, 1),
            )
            r4_leads, r4_hands, r4_score = next_round(
                current_hands=r3_hands, leads=r3_leads, game_round=4, game_score=r3_score
            )
            r5_leads, r5_hands, r5_score = next_round(
                current_hands=r4_hands, leads=r4_leads, game_round=5, game_score=r4_score
            )
            results = r5_score.reshape(r5_score.shape[0], 5)
        
        elif tricks == 3:
            # 3 tricks: Simulate rounds 4, 5
            r3_leads, r3_hands, r3_score = next_round(
                current_hands=filtered_hands,
                leads=filtered_leads,
                game_round=4,
                game_score=filtered_leads.reshape(-1, 1),
            )
            r4_leads, r4_hands, r4_score = next_round(
                current_hands=r3_hands, leads=r3_leads, game_round=5, game_score=r3_score
            )
            results = r4_score.reshape(r4_score.shape[0], 5)
        
        elif tricks == 2:
            # 2 tricks: Simulate round 5 only
            r3_leads, r3_hands, r3_score = next_round(
                current_hands=filtered_hands,
                leads=filtered_leads,
                game_round=5,
                game_score=filtered_leads.reshape(-1, 1),
            )
            results = r3_score.reshape(r3_score.shape[0], 5)

        elif tricks == 1:
            # 1 trick: No more rounds to simulate
            results = filtered_leads.reshape(-1, 1)

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

        prune_masks[pruned_idx] = mask
        best_pruned_branch_res[pruned_idx] = np.mean(meta_results)
        pruned_idx += 1

    # Select best branch based on which team the responder is on
    if player % 2 == 0:
        best_prune = prune_masks[np.argmax(best_pruned_branch_res)]
    else: 
        best_prune = prune_masks[np.argmin(best_pruned_branch_res)]

    final_prune = given_hands[best_prune]
    leads_prune = given_leads[best_prune]

    return final_prune, leads_prune

def find_best_response(
    lead: int,
    hands: np.ndarray,
    tricks: int,
    previous_winners: np.ndarray,
    best_opener: int,
):
    pr1, pr1_leads = player_best_prune(
        player=(lead + 1) % 4,
        first_response=True,
        starting_hands=hands,
        best_opener=best_opener,
        starting_lead=lead,
        tricks=tricks,
        previous_winners=previous_winners,
    )

    pr2, pr2_leads = player_best_prune(
        player=(lead + 2) % 4,
        first_response=False,
        best_opener=best_opener,
        starting_lead=lead,
        tricks=tricks,
        given_hands=pr1,
        given_leads=pr1_leads,
    )

    pr3, pr3_leads = player_best_prune(
        player=(lead + 3) % 4,
        first_response=False,
        best_opener=best_opener,
        starting_lead=lead,
        tricks=tricks,
        given_hands=pr2,
        given_leads=pr2_leads,
    )

    pruned_hand = pr3[0]
    pruned_lead = pr3_leads[0]

    if pruned_lead.size != 1:
        print(pruned_hand)
        raise ValueError("Pruning did not result in a single optimal branch. Check the pruning logic.")

    return pruned_hand, pruned_lead


# def definitive_winner(given_hands: np.ndarray, verbose: bool):
#     """
#     Determines the definitive winner of a complete 5-trick Euchre game through optimal play.
    
#     This function simulates a full game of Euchre where both teams play optimally at each 
#     decision point. It uses a minimax-style approach, alternating between finding the best 
#     opening card for the leading team and the best response from the opposing team for all 
#     five tricks.
    
#     The game proceeds as follows:
#     1. Player 1 (odd team) finds their optimal opening card
#     2. The opposing team's best response is calculated
#     3. The winner of trick 1 leads trick 2, and the process repeats
#     4. This continues for all 5 tricks
#     5. The team that wins 3+ tricks is declared the winner
    
#     Arguments:
#         given_hands (np.ndarray): A 3D array of shape (4, 5, 2) representing the initial 
#             hands dealt to all four players. Each card is represented as a 2D vector [x, y].
#             Players are indexed 0-3, where:
#                 - Team 0 (even): players 0 and 2
#                 - Team 1 (odd): players 1 and 3
#         verbose (bool): If True, prints detailed information about each trick including 
#             cards played, win probabilities, and intermediate game states. If False, runs 
#             silently.
    
#     Returns:
#         int: The winning team index:
#             - 0 if the odd team (players 1, 3) won the game (3+ tricks)
#             - 1 if the even team (players 0, 2) won the game (3+ tricks)
#     """

#     best_opener = find_best_opener(
#         hands=given_hands, lead=1, tricks=5, previous_winners=np.array([]), sim_func=n_trick_sim, verbose=verbose
#     )
#     r2_optimal, winner_round1 = find_best_response(
#         hands=given_hands,
#         lead=1,
#         tricks=5,
#         previous_winners=np.array([]),
#         best_opener=best_opener,
#         verbose=verbose
#     )
#     r2_best_opener = find_best_opener(
#         hands=r2_optimal,
#         lead=winner_round1,
#         tricks=4,
#         previous_winners=np.array([winner_round1]),
#         sim_func=n_trick_sim,
#         verbose=verbose
#     )
#     r3_optimal, winner_round2 = find_best_response(
#         hands=r2_optimal,
#         lead=winner_round1,
#         tricks=4,
#         previous_winners=np.array([winner_round1]),
#         best_opener=r2_best_opener,
#         verbose=verbose

#     )
#     r3_best_opener = find_best_opener(
#         hands=r3_optimal,
#         lead=winner_round2,
#         tricks=3,
#         previous_winners=np.array([winner_round1, winner_round2]),
#         sim_func=n_trick_sim,
#         verbose=verbose
#     )
#     r4_optimal, winner_round3 = find_best_response(
#         hands=r3_optimal,
#         lead=winner_round2,
#         tricks=3,
#         previous_winners=np.array([winner_round1, winner_round2]),
#         best_opener=r3_best_opener,
#         verbose=verbose

#     )
#     r4_best_opener = find_best_opener(
#         hands=r4_optimal,
#         lead=winner_round3,
#         tricks=2,
#         previous_winners=np.array([winner_round1, winner_round2, winner_round3]),
#         sim_func=n_trick_sim,
#         verbose=verbose
#     )
#     r5_optimal, winner_round4 = find_best_response(
#         hands=r4_optimal,
#         lead=winner_round3,
#         tricks=2,
#         previous_winners=np.array([winner_round1, winner_round2, winner_round3]),
#         best_opener=r4_best_opener,
#         verbose=verbose

#     )
#     r5_best_opener = find_best_opener(
#         hands=r5_optimal,
#         lead=winner_round4,
#         tricks=1,
#         previous_winners=np.array(
#             [winner_round1, winner_round2, winner_round3, winner_round4]
#         ),
#         sim_func=n_trick_sim,
#         verbose=verbose
#     )

#     r6_optimal, winner_round5 = find_best_response(
#         hands=r5_optimal,
#         lead=winner_round4,
#         tricks=1,
#         previous_winners=np.array([winner_round1, winner_round2, winner_round3, winner_round4]),
#         best_opener=r5_best_opener,
#         verbose=verbose

#     )

#     total_winners = np.array([winner_round1, winner_round2, winner_round3, winner_round4, winner_round5], dtype=np.int64)

#     result = declare_winner(winners=total_winners)

#     return result

    

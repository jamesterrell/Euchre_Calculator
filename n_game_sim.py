from n_play_round import round1, next_round
from numba import njit
import numpy as np


@njit
def n_game_sim(game_hand: np.ndarray, eval_position: int):
    """
    Simulates a full game using the provided hands and evaluates the outcome based on
    a specified player's position. The function runs five rounds of play, tracking the
    results and calculating the final score based on the player's performance. It then
    returns the proportion of games in which the player's cards results in a win.

    Arguments:
        game_hand (numpy.ndarray): A 3D array representing the cards dealt to each player.
        eval_position (int): The index of the player whose cards are being evaluated.

    Returns:
        float: The mean performance evaluation for the specified player, indicating the
        proportion of games where the player's performance exceeds a threshold.
    """
    r2_leads, r2_hands = round1(hands_dealt=game_hand)
    r3_leads, r3_hands, r2_score = next_round(
        current_hands=r2_hands,
        leads=r2_leads,
        game_round=2,
        game_score=r2_leads.reshape(-1, 1),
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

    if eval_position % 2 == 0:
        for i in range(len(results)):
            if np.sum(results[i] % 2) == 0:
                score = 2
            elif np.sum(results[i] % 2) == 1 or np.sum(results[i] % 2) == 2:
                score = 1
            elif (
                np.sum(results[i] % 2) == 3
                or np.sum(results[i] % 2) == 4
                or np.sum(results[i] % 2) == 4
            ):
                score = 0
            meta_results[i] = score
    else:
        for i in range(len(results)):
            if np.sum(results[i] % 2) == 5:
                score = 2
            elif np.sum(results[i] % 2) == 4 or np.sum(results[i] % 2) == 5:
                score = 1
            elif (
                np.sum(results[i] % 2) == 2
                or np.sum(results[i] % 2) == 1
                or np.sum(results[i] % 2) == 0
            ):
                score = 0
            meta_results[i] = score

    return np.mean(meta_results)


@njit(parallel=True, fastmath=True)
def meta_game_sim(meta_hands: list, eval_position):
    """
    Simulates a series of full games using the provided hands and evaluates the performance
    of a specified player across all games. The function runs multiple simulations of `n_game_sim`,
    calculating the proportion of games where the player's cards results in a win.

    Arguments:
        meta_hands (list): A list of 3D arrays, each representing a set of cards dealt to each player.
        eval_position (int): The index of the player whose performance is being evaluated.

    Returns:
        float: The mean evaluation across all games, indicating the proportion of games where
        the specified player's cards results in a win.
    """
    meta_results = np.zeros(len(meta_hands), dtype=np.int64)
    for i in range(len(meta_hands)):
        res = n_game_sim(game_hand=meta_hands[i], eval_position=eval_position)
        meta_results[i] = res

    return np.mean(meta_results)

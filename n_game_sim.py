from n_play_round import round1, next_round
from numba import njit
import numpy as np

@njit
def n_game_sim(game_hand: np.ndarray, eval_position: int):
    r2_leads, r2_hands = round1(hands_dealt=game_hand)
    r3_leads, r3_hands, r2_score = next_round(current_hands=r2_hands, leads=r2_leads, game_round=2, game_score=r2_leads.reshape(-1, 1))
    r4_leads, r4_hands, r3_score = next_round(current_hands=r3_hands, leads=r3_leads, game_round=3, game_score=r2_score)
    r5_leads, r5_hands, r4_score = next_round(current_hands=r4_hands, leads=r4_leads, game_round=4, game_score=r3_score)
    r6_leads, r6_hands, r5_score = next_round(current_hands=r5_hands, leads=r5_leads, game_round=5, game_score=r4_score)
    results = r5_score.reshape(r5_score.shape[0], 5)

    meta_results = np.zeros(results.shape[0], dtype=np.bool_)
    
    if eval_position % 2 == 0:
        for i in range(len(results)):
            meta_results[i] = np.sum(results[i]%2)<3
    else:
        for i in range(len(results)):
            meta_results[i] = np.sum(results[i]%2)>=3

    return np.mean(meta_results)


@njit(parallel=True, fastmath=True, nogil=True)
def meta_game_sim(meta_hands: list, eval_position):
    meta_results = np.zeros(len(meta_hands), dtype=np.bool_)
    for i in range(len(meta_hands)):
        res = n_game_sim(game_hand=meta_hands[i], eval_position=eval_position)
        meta_results[i] = res>0.5

    
    return np.mean(meta_results)

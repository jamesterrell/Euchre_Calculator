from n_game_sim import n_game_sim
from n_play_round import round1, next_round
import numpy as np
from numba import njit

@njit
def find_best_opener(hands):
    winning_chances = np.zeros(5)
    for i in range(5):
       winning_chances[i] = n_game_sim(game_hand=hands, eval_position=0, r1_chosen_card=i)

    print(winning_chances)
    return np.argmax(winning_chances)


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
                meta_results[i] = np.sum(results[i]%2)<3
        else:
            for i in range(len(results)):
                meta_results[i] = np.sum(results[i]%2)>=3

        resp_win_chances[counter] = np.mean(meta_results)
        counter +=1

    print(resp_win_chances)
    best_response = np.argmax(resp_win_chances)
    return r2_hands[best_response], r2_leads[best_response]
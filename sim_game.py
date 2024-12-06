import numpy as np
from play_round import play_round, find_winners_vectorized
from branch_calc import compute_set_difference
from numba import jit

@jit(nopython=True)
def game_sim(dealt_hands: list, first_play: range | list = range(5), second_play: range | list = range(4) ):
    # round1
    score = []
    round1_tricks = []

    for i in first_play:
        round1_play = play_round(hands=dealt_hands, lead=0, card_play=i)
        round1_tricks.append(round1_play)
        score.append(
            find_winners_vectorized(
                leads=np.zeros(len(round1_play), dtype=int), tricks=round1_play
            )
        )
    score = np.concatenate(score)
    round1_tricks = np.concatenate(round1_tricks)
    round2_hands = compute_set_difference(
        len(round1_tricks) * [dealt_hands], round1_tricks
    )

    print(len(round1_tricks))

    # round2
    round2_score = []
    round2_tricks = []
    round3_hands = []
    r2_full_res = []

    for i, j in zip(round2_hands, score):
        for k in second_play:
            round2_play = play_round(hands=i, lead=j, card_play=k)
            round2_tricks.append(round2_play)
            winners = find_winners_vectorized(
                leads=len(round2_play) * [j], tricks=round2_play
            )
            round2_score.append(winners)
            full_score = np.column_stack((len(round2_play) * [j], winners))
            r2_full_res.append(full_score)
            test = len(round2_play) * [i]
            round3_hands.append(test)
    round2_score = np.concatenate(round2_score)
    round2_tricks = np.concatenate(round2_tricks)
    round3_hands = np.concatenate(round3_hands)
    round3_hands_comp = compute_set_difference(round3_hands, round2_tricks)
    r2_full_res = np.concatenate(r2_full_res)

    print(len(round2_tricks))

    # round3

    round3_score = []
    round3_tricks = []
    round4_hands = []
    r3_full_res = []

    for h, i, j in zip(r2_full_res, round3_hands_comp, round2_score):
        for k in range(3):
            round3_play = play_round(hands=i, lead=j, card_play=k)
            round3_tricks.append(round3_play)
            winners = find_winners_vectorized(
                leads=len(round3_play) * [j], tricks=round3_play
            )
            round3_score.append(winners)
            full_score = np.column_stack((len(round3_play) * [h], winners))
            r3_full_res.append(full_score)

            test = len(round3_play) * [i]
            round4_hands.append(test)

    round3_score = np.concatenate(round3_score)
    r3_full_res = np.concatenate(r3_full_res)
    round3_tricks = np.concatenate(round3_tricks)
    round4_hands = np.concatenate(round4_hands)
    round4_hands_comp = compute_set_difference(round4_hands, round3_tricks)

    print(len(round3_tricks))

    round4_score = []
    round4_tricks = []
    round5_hands = []
    r4_full_res = []

    for h, i, j in zip(r3_full_res, round4_hands_comp, round3_score):
        for k in range(2):
            round4_play = play_round(hands=i, lead=j, card_play=k)
            round4_tricks.append(round4_play)
            winners = find_winners_vectorized(
                leads=len(round4_play) * [j], tricks=round4_play
            )
            round4_score.append(winners)
            full_score = np.column_stack((len(round4_play) * [h], winners))
            r4_full_res.append(full_score)

            test = len(round4_play) * [i]
            round5_hands.append(test)

    round4_score = np.concatenate(round4_score)
    r4_full_res = np.concatenate(r4_full_res)
    round4_tricks = np.concatenate(round4_tricks)
    round5_hands = np.concatenate(round5_hands)
    round5_hands_comp = compute_set_difference(round5_hands, round4_tricks)

    print(len(round4_tricks))

    round5_score = []

    for h, i, j in zip(r4_full_res, round5_hands_comp, round4_score):
        round5_play = play_round(hands=i, lead=j, card_play=0)
        winners = find_winners_vectorized(
            leads=len(round5_play) * [j], tricks=round5_play
        )
        full_score = np.column_stack((len(round5_play) * [h], winners))
        round5_score.append(full_score)

    round5_score = np.concatenate(round5_score)

    winning_teams = [res % 2 for res in round5_score]

    return np.mean([np.sum(win) > 2 for win in winning_teams])

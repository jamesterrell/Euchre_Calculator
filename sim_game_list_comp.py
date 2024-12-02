from play_round import PlayRound, find_winners_vectorized
from branch_calc import compute_set_difference
import numpy as np

#something is a bit off here, not sure what though.

def lc_game_sim(dealt_hands: list, first_play: range | list = range(5)):
    # round 1
    round1 = [PlayRound(hands=dealt_hands, lead=0, card_play=i) for i in first_play]
    round1_tricks = np.concatenate([trick.play_round() for trick in round1])
    round1_hands = np.concatenate([trick.show_hands() for trick in round1])
    score = find_winners_vectorized(
        leads=np.zeros(len(round1_tricks), dtype=int), tricks=round1_tricks
    )

    round2_hands = compute_set_difference(round1_hands, round1_tricks)

    print(len(score))

    # round 2
    round2 = [
        PlayRound(hands=hands, lead=score[i], card_play=j)
        for i, hands in enumerate(round2_hands)
        for j in range(4)  # Assuming 4 possible card plays in round 2
    ]

    # Generate all possible tricks for round 2
    round2_tricks = np.concatenate([trick.play_round() for trick in round2])
    round2_hands = np.concatenate([trick.show_hands() for trick in round2])
    # Find winners with leads based on round 1 scores
    rd2_score = find_winners_vectorized(
        leads=np.repeat(score, 4),  # Repeat leads to match number of tricks
        tricks=round2_tricks,
    )
    round3_hands = compute_set_difference(round2_hands, round2_tricks)

    print(len(rd2_score))

    # round 3
    round3 = [
        PlayRound(hands=hands, lead=rd2_score[i], card_play=j)
        for i, hands in enumerate(round3_hands)
        for j in range(3)
    ]

    round3_tricks = np.concatenate([trick.play_round() for trick in round3])
    round3_hands = np.concatenate([trick.show_hands() for trick in round3])
    rd3_score = find_winners_vectorized(leads=np.repeat(rd2_score, 3), tricks=round3_tricks)

    round4_hands = compute_set_difference(round3_hands, round3_tricks)

    print(len(rd3_score))

    # round 4

    round4 = [
        PlayRound(hands=hands, lead=rd3_score[i], card_play=j)
        for i, hands in enumerate(round4_hands)
        for j in range(2)
    ]

    round4_tricks = np.concatenate([trick.play_round() for trick in round4])
    round4_hands = np.concatenate([trick.show_hands() for trick in round4])
    rd4_score = find_winners_vectorized(leads=np.repeat(rd3_score, 2), tricks=round4_tricks)
    round5_hands = compute_set_difference(round4_hands, round4_tricks)

    print(len(rd4_score))

    # # round 5
    round5 = round1 = [PlayRound(hands=i, lead=j, card_play=0) for i, j in zip(round5_hands, rd4_score)]

    round5_tricks = np.concatenate([trick.play_round() for trick in round5])
    rd5_score = find_winners_vectorized(leads=rd4_score, tricks=round5_tricks)

    return len(rd5_score)
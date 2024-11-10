import numpy as np

# hearts = negative x axis coords
# diamonds = positive x axis coords
# spades = positive y axis coords
# clubs = negative y axis coords
full_euchre_deck = np.array(
    [
        [-9, 0],
        [-10, 0],
        [-11, 0],
        [-12, 0],
        [-13, 0],
        [-14, 0],
        [0, -9],
        [0, -10],
        [0, -11],
        [0, -12],
        [0, -13],
        [0, -14],
        [9, 0],
        [10, 0],
        [11, 0],
        [12, 0],
        [13, 0],
        [14, 0],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
    ]
)


def card_evaluator(card: np.array):
    evaluator = np.array([0, 0])
    return np.linalg.norm(card - evaluator)

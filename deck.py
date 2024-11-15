import numpy as np

# hearts = negative x axis coords
# diamonds = positive x axis coords
# spades = positive y axis coords
# clubs = negative y axis coords

# for the purposes of this program, we'll consider spades to always be trump,
# which is why it is getting a higher value
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
        [0, -135],
        [9, 0],
        [10, 0],
        [11, 0],
        [12, 0],
        [13, 0],
        [14, 0],
        [0, 90],
        [0, 100],
        [0, 110],
        [0, 120],
        [0, 130],
        [0, 140],
    ]
)

